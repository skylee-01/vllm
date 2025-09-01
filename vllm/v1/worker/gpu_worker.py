# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A GPU worker class."""
import copy
import gc
import os
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)
from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.model_executor.warmup.kernel_warmup import kernel_warmup
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.utils import GiB_bytes, MemorySnapshot, memory_profiling
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, DraftTokenIds,
                             ModelRunnerOutput)
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.worker_base import WorkerBase

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.core.sched.output import SchedulerOutput


class Worker(WorkerBase):
    """一个在GPU上执行模型推理的worker。

    这个类负责初始化GPU设备，加载模型，管理KV缓存，以及执行模型的前向传播。
    它还处理分布式环境的设置和模型配置的更新。
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        """初始化GPU Worker。

        Args:
            vllm_config: vLLM的配置对象。
            local_rank: 当前Worker在本地（单个节点内）的GPU排名。
            rank: 当前Worker在整个分布式环境中的全局排名。
            distributed_init_method: 分布式初始化方法。
            is_driver_worker: 是否是驱动Worker，驱动Worker负责初始化模型并行组。
        """

        super().__init__(vllm_config=vllm_config,
                         local_rank=local_rank,
                         rank=rank,
                         distributed_init_method=distributed_init_method,
                         is_driver_worker=is_driver_worker)

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        # Buffers saved before sleep
        self._sleep_saved_buffers: dict[str, torch.Tensor] = {}

        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        torch_profiler_trace_dir)
            logger.debug(
                "Profiler config: record_shapes=%s,"
                "profile_memory=%s,with_stack=%s,with_flops=%s",
                envs.VLLM_TORCH_PROFILER_RECORD_SHAPES,
                envs.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY,
                envs.VLLM_TORCH_PROFILER_WITH_STACK,
                envs.VLLM_TORCH_PROFILER_WITH_FLOPS,
            )
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=envs.VLLM_TORCH_PROFILER_RECORD_SHAPES,
                profile_memory=envs.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY,
                with_stack=envs.VLLM_TORCH_PROFILER_WITH_STACK,
                with_flops=envs.VLLM_TORCH_PROFILER_WITH_FLOPS,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir, use_gzip=True))
        else:
            self.profiler = None

    def sleep(self, level: int = 1) -> None:
        """让Worker进入睡眠模式以释放GPU内存。

        在睡眠模式下，根据指定的级别，部分或全部GPU内存会被释放，
        以供其他进程使用。级别2会保存模型权重到CPU，以便之后恢复。

        Args:
            level: 睡眠级别。1表示释放KV缓存，2表示释放KV缓存并保存模型权重。
        """
        from vllm.device_allocator.cumem import CuMemAllocator

        free_bytes_before_sleep = torch.cuda.mem_get_info()[0]

        # Save the buffers before level 2 sleep
        if level == 2:
            model = self.model_runner.model
            self._sleep_saved_buffers = {
                name: buffer.cpu().clone()
                for name, buffer in model.named_buffers()
            }

        allocator = CuMemAllocator.get_instance()
        allocator.sleep(offload_tags=("weights", ) if level == 1 else tuple())
        free_bytes_after_sleep, total = torch.cuda.mem_get_info()
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
            "Sleep mode freed %.2f GiB memory, "
            "%.2f GiB memory is still in use.", freed_bytes / GiB_bytes,
            used_bytes / GiB_bytes)

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        """唤醒Worker并恢复GPU内存。

        此方法将恢复之前由 `sleep` 方法释放的GPU内存。
        如果模型权重在睡眠期间被保存到CPU，它们也会被恢复到GPU。

        Args:
            tags: 可选的标签列表，用于选择性地唤醒特定类型的内存池。
        """
        from vllm.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags)

        # Restore the buffers after level 2 sleep
        if len(self._sleep_saved_buffers):
            model = self.model_runner.model
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}

    def _maybe_get_memory_pool_context(self,
                                       tag: str) -> AbstractContextManager:
        """根据是否启用睡眠模式，返回一个内存池上下文管理器。

        如果启用了睡眠模式，并且标签为“weights”，它会断言当前内存使用量为0，
        然后返回一个CuMemAllocator的内存池上下文。
        否则，返回一个空的上下文管理器。

        Args:
            tag: 内存池的标签，例如“weights”或“kv_cache”。

        Returns:
            一个AbstractContextManager实例，用于管理内存池。
        """
        if self.vllm_config.model_config.enable_sleep_mode:
            from vllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            if tag == "weights":
                assert allocator.get_current_usage() == 0, (
                    "Sleep mode can only be "
                    "used for one instance per process.")
            context = allocator.use_memory_pool(tag=tag)
        else:
            context = nullcontext()
        return context

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """初始化KV缓存配置中的GPU和CPU块数量。

        Args:
            num_gpu_blocks: 可用于KV缓存的GPU块数量。
            num_cpu_blocks: 可用于KV缓存的CPU块数量。
        """
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def init_device(self):
        """初始化GPU设备和分布式环境。

        此方法设置CUDA设备，检查GPU是否支持模型的dtype，
        并初始化分布式训练环境（如NCCL）。
        它还会进行内存快照，以确保有足够的GPU内存用于KV缓存。
        """
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            current_platform.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            gc.collect()
            torch.cuda.empty_cache()

            # take current memory snapshot
            self.init_snapshot = MemorySnapshot()
            self.requested_memory = (self.init_snapshot.total_memory *
                                     self.cache_config.gpu_memory_utilization)
            if self.init_snapshot.free_memory < self.requested_memory:
                GiB = lambda b: round(b / GiB_bytes, 2)
                raise ValueError(
                    f"Free memory on device "
                    f"({GiB(self.init_snapshot.free_memory)}/"
                    f"{GiB(self.init_snapshot.total_memory)} GiB) on startup "
                    f"is less than desired GPU memory utilization "
                    f"({self.cache_config.gpu_memory_utilization}, "
                    f"{GiB(self.requested_memory)} GiB). Decrease GPU memory "
                    f"utilization or reduce GPU memory used by other processes."
                )
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.vllm_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank,
                                            current_platform.dist_backend)
        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Construct the model runner
        self.model_runner: GPUModelRunner = GPUModelRunner(
            self.vllm_config, self.device)

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

    # FIXME(youkaichao & ywang96): Use TorchDispatchMode instead of memory pool
    # to hijack tensor allocation.
    def load_model(self) -> None:
        """加载模型到GPU内存。

        如果启用了睡眠模式，模型权重会在内存池的上下文中加载。
        """
        eep_scale_up = os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH") == "1"
        with self._maybe_get_memory_pool_context(tag="weights"):
            self.model_runner.load_model(eep_scale_up=eep_scale_up)

    def update_config(self, overrides: dict[str, Any]) -> None:
        """更新模型运行器的配置。

        Args:
            overrides: 包含要更新配置的字典。
        """
        self.model_runner.update_config(overrides)

    def reload_weights(self) -> None:
        """重新加载模型的权重。

        此方法通常用于在不重新初始化整个模型的情况下更新模型参数，
        例如在LoRA适配器切换时。
        """
        with self._maybe_get_memory_pool_context(tag="weights"):
            self.model_runner.reload_weights()

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """分析模型峰值内存使用量，以确定可用于KV缓存的内存大小。

        引擎首先对现有内存使用情况进行分析。
        然后，计算可用于KV缓存的字节数。

        提示:
            您可以通过调整 `gpu_memory_utilization` 参数来限制GPU内存的使用。

        Returns:
            可用于KV缓存的内存字节数。
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        GiB = lambda b: b / GiB_bytes

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        with memory_profiling(
                self.init_snapshot,
                weights_memory=int(
                    self.model_runner.model_memory_usage)) as profile_result:
            self.model_runner.profile_run()

        free_gpu_memory = profile_result.after_profile.free_memory
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        assert self.init_snapshot.free_memory > free_gpu_memory, (
            "Error in memory profiling. "
            f"Initial free memory {GiB(self.init_snapshot.free_memory)} GiB, "
            f"current free memory {GiB(free_gpu_memory)} GiB. "
            "This happens when other processes sharing the same container "
            "release GPU memory while vLLM is profiling during initialization. "
            "To fix this, ensure consistent GPU memory allocation or "
            "isolate vLLM in its own container.")
        available_kv_cache_memory = self.requested_memory \
            - profile_result.non_kv_cache_memory

        unrequested_memory = self.init_snapshot.free_memory \
            - self.requested_memory
        logger.debug(
            "Initial free memory: %.2f GiB; "
            "Requested memory: %.2f (util), %.2f GiB",
            GiB(self.init_snapshot.free_memory),
            self.cache_config.gpu_memory_utilization,
            GiB(self.requested_memory),
        )
        logger.debug(
            "Free memory after profiling: %.2f GiB (total), "
            "%.2f GiB (within requested)",
            GiB(free_gpu_memory),
            GiB(free_gpu_memory - unrequested_memory),
        )
        logger.debug(profile_result)
        logger.info("Available KV cache memory: %.2f GiB",
                    GiB(available_kv_cache_memory))
        gc.collect()

        return int(available_kv_cache_memory)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """获取KV缓存的规范。

        Returns:
            一个字典，包含KV缓存的规范。
        """
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """使用指定的KV缓存配置分配GPU KV缓存。

        Args:
            kv_cache_config: KV缓存的配置对象。
        """

        if self.vllm_config.model_config.enable_sleep_mode:
            from vllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            context = allocator.use_memory_pool(tag="kv_cache")
        else:
            from contextlib import nullcontext
            context = nullcontext()
        with context:
            self.model_runner.initialize_kv_cache(kv_cache_config)

    def compile_or_warm_up_model(self) -> None:
        """编译或预热模型。

        此方法用于在模型执行之前编译或预热模型，以优化性能。
        它会针对不同的批处理大小进行模型预热，并捕捉CUDA图（如果未强制使用eager模式）。
        此外，还会预热采样器并预分配内存缓冲区，以避免内存碎片问题。
        """
        warmup_sizes = self.vllm_config.compilation_config.compile_sizes.copy()
        if not self.model_config.enforce_eager:
            warmup_sizes = [
                x for x in warmup_sizes if x not in
                self.vllm_config.compilation_config.cudagraph_capture_sizes
            ]
        # We skip EPLB here since we don't want to record dummy metrics
        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(size, skip_eplb=True)

        if not self.model_config.enforce_eager:
            self.model_runner.capture_model()

        # Warm up sampler and preallocate memory buffer for logits and other
        # sampling related tensors of max possible shape to avoid memory
        # fragmentation issue.
        # NOTE: This is called after `capture_model` on purpose to prevent
        # memory buffers from being cleared by `torch.cuda.empty_cache`.
        if get_pp_group().is_last_rank:
            max_num_reqs = min(self.scheduler_config.max_num_seqs,
                               self.scheduler_config.max_num_batched_tokens)

            # We skip EPLB here since we don't want to record dummy metrics
            hidden_states, last_hidden_states = \
                self.model_runner._dummy_run(
                    num_tokens=max_num_reqs,
                    skip_eplb=True,
                )
            if self.model_runner.is_pooling_model:
                self.model_runner._dummy_pooler_run(hidden_states)
            else:
                self.model_runner._dummy_sampler_run(
                    hidden_states=last_hidden_states)

        # Warmup kernels used during model execution
        kernel_warmup(self)

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    def get_model(self) -> nn.Module:
        """获取底层的PyTorch模型实例。

        Returns:
            PyTorch模型实例。
        """
        return self.model_runner.get_model()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        """获取模型支持的任务列表。

        Returns:
            一个包含模型支持任务的元组。
        """
        return self.model_runner.get_supported_tasks()

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        """执行模型的前向传播。

        此方法接收调度器输出，执行模型推理，并处理中间张量的传输（如果存在流水线并行）。

        Args:
            scheduler_output: 调度器生成的输出，包含要处理的请求信息。

        Returns:
            模型运行器的输出，可能包含生成的token、KV缓存更新等，或者为None。
        """
        intermediate_tensors = None
        if not get_pp_group().is_first_rank:
            intermediate_tensors = IntermediateTensors(
                get_pp_group().recv_tensor_dict(
                    all_gather_group=get_tp_group()))

        output = self.model_runner.execute_model(scheduler_output,
                                                 intermediate_tensors)

        parallel_config = self.vllm_config.parallel_config
        if parallel_config.distributed_executor_backend != "external_launcher" \
            and not get_pp_group().is_last_rank:
            assert isinstance(output, IntermediateTensors)
            get_pp_group().send_tensor_dict(output.tensors,
                                            all_gather_group=get_tp_group())

            kv_connector_output = output.kv_connector_output
            if not kv_connector_output:
                return None

            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            if (not kv_connector_output.finished_sending
                    and not kv_connector_output.finished_recving):
                return EMPTY_MODEL_RUNNER_OUTPUT

            output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return output

        assert isinstance(output, ModelRunnerOutput)
        return output

    def take_draft_token_ids(self) -> Optional[DraftTokenIds]:
        """从模型运行器中获取草稿token ID。

        此方法用于推测解码，获取模型生成的草稿token，以便进行验证。

        Returns:
            一个包含草稿token ID的DraftTokenIds对象，如果不可用则为None。
        """
        return self.model_runner.take_draft_token_ids()

    def profile(self, is_start: bool = True):
        """启动或停止Torch Profiler。

        此方法用于控制PyTorch性能分析器的启动和停止，
        并可以在停止时打印性能统计信息。

        Args:
            is_start: 如果为True，则启动profiler；如果为False，则停止profiler并打印结果。
        """
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()
            print(self.profiler.key_averages().table(
                sort_by="self_cuda_time_total"))

    def execute_dummy_batch(self) -> None:
        """执行一个虚拟批次的前向传播。

        此方法用于在不进行实际推理的情况下运行模型，
        主要用于模型预热或测试目的。
        """
        self.model_runner._dummy_run(1)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        """添加一个LoRA适配器到模型。

        Args:
            lora_request: 包含LoRA适配器信息的请求对象。

        Returns:
            如果成功添加LoRA，则返回True，否则返回False。
        """
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        """从模型中移除一个LoRA适配器。

        Args:
            lora_id: 要移除的LoRA适配器的唯一标识符。

        Returns:
            如果成功移除LoRA，则返回True，否则返回False。
        """
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        """列出当前加载的所有LoRA适配器的ID。

        Returns:
            一个包含所有LoRA ID的集合。
        """
        return self.model_runner.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        """将指定的LoRA适配器固定在内存中。

        Args:
            lora_id: 要固定的LoRA适配器的唯一标识符。

        Returns:
            如果成功固定LoRA，则返回True，否则返回False。
        """
        return self.model_runner.pin_lora(lora_id)

    def check_health(self) -> None:
        """检查Worker的健康状态。

        当前实现中，Worker只要正在运行就被认为是健康的。
        """
        return

    def _eplb_before_scale_down(self, old_ep_size: int,
                                new_ep_size: int) -> None:
        """在弹性并行负载均衡 (EPLB) 缩小规模之前重新分配专家。

        此方法在EP组规模缩小之前调用，以确保专家负载能够正确重新分配。

        Args:
            old_ep_size: 旧的专家并行组大小。
            new_ep_size: 新的专家并行组大小。
        """
        from vllm.distributed.parallel_state import get_ep_group
        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Starting expert resharding "
                        "before scaling down...")
        rank_mapping = {
            old_ep_rank: old_ep_rank if old_ep_rank < new_ep_size else -1
            for old_ep_rank in range(old_ep_size)
        }
        assert self.model_runner.eplb_state is not None
        self.model_runner.eplb_state.rearrange(self.model_runner.model,
                                               execute_shuffle=True,
                                               global_expert_load=None,
                                               rank_mapping=rank_mapping)
        torch.cuda.synchronize()
        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Expert resharding completed!")

    def _eplb_after_scale_up(
            self, old_ep_size: int, new_ep_size: int,
            global_expert_load: Optional[torch.Tensor]) -> None:
        """在弹性并行负载均衡 (EPLB) 扩大规模之后重新分配专家。

        此方法在EP组规模扩大之后调用，以确保专家负载能够正确重新分配。

        Args:
            old_ep_size: 旧的专家并行组大小。
            new_ep_size: 新的专家并行组大小。
            global_expert_load: 全局专家负载张量。
        """
        from vllm.distributed.parallel_state import get_ep_group
        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Starting expert resharding "
                        "after scaling up...")
        rank_mapping = {
            old_ep_rank: old_ep_rank
            for old_ep_rank in range(old_ep_size)
        }
        assert self.model_runner.eplb_state is not None
        self.model_runner.eplb_state.rearrange(
            self.model_runner.model,
            execute_shuffle=True,
            global_expert_load=global_expert_load,
            rank_mapping=rank_mapping)
        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Expert resharding completed!")

    def _reconfigure_parallel_config(
            self, reconfig_request: ReconfigureDistributedRequest) -> None:
        """
        使用提供的重新配置请求更新并行配置。

        Args:
            reconfig_request: 包含新的并行配置信息的请求对象。
        """
        parallel_config = self.vllm_config.parallel_config
        parallel_config.data_parallel_size = \
            reconfig_request.new_data_parallel_size
        if reconfig_request.new_data_parallel_rank != \
        ReconfigureRankType.KEEP_CURRENT_RANK:
            parallel_config.data_parallel_rank = \
                reconfig_request.new_data_parallel_rank
        if reconfig_request.new_data_parallel_rank_local != \
        ReconfigureRankType.KEEP_CURRENT_RANK:
            parallel_config.data_parallel_rank_local = \
                reconfig_request.new_data_parallel_rank_local
        parallel_config.data_parallel_master_ip = \
            reconfig_request.new_data_parallel_master_ip
        parallel_config.data_parallel_master_port = \
            reconfig_request.new_data_parallel_master_port

    def _reconfigure_moe(self, old_ep_size: int,
                         new_ep_size: int) -> Optional[torch.Tensor]:
        """
        使用提供的重新配置请求重新配置MoE模块。

        此方法会更新MoE模块中的专家数量和并行配置。
        在规模扩大时，会返回全局专家负载；在规模缩小时，返回None。

        Args:
            old_ep_size: 旧的专家并行组大小。
            new_ep_size: 新的专家并行组大小。

        Returns:
            如果 new_ep_size > old_ep_size，则返回全局专家负载，否则返回None。
        """
        from vllm.distributed.parallel_state import (
            get_dp_group, get_ep_group, prepare_communication_buffer_for_model)
        from vllm.model_executor.layers.fused_moe.layer import (
            FusedMoEParallelConfig)

        parallel_config = self.vllm_config.parallel_config
        moe_modules = [
            module for module in self.model_runner.model.modules()
            if module.__class__.__name__ == "FusedMoE"
        ]
        num_local_experts = moe_modules[0].moe_config.num_local_experts
        assert all(module.moe_config.num_local_experts == num_local_experts
                   for module in moe_modules), (
                       "All MoE modules must have the same number of experts")
        for module in moe_modules:
            module.moe_config.num_experts = num_local_experts * new_ep_size
            module.global_num_experts = module.moe_config.num_experts
            module.moe_parallel_config = FusedMoEParallelConfig.make(
                tp_size_=get_tp_group().world_size,
                dp_size_=get_dp_group().world_size,
                vllm_parallel_config=parallel_config,
            )
            module.moe_config.moe_parallel_config = module.moe_parallel_config
        if new_ep_size < old_ep_size:
            num_local_physical_experts = num_local_experts
            assert self.model_runner.eplb_state is not None
            new_physical_experts = \
                self.model_runner.eplb_state.physical_to_logical_map.shape[1]
            parallel_config.num_redundant_experts = (
                new_physical_experts -
                self.model_runner.eplb_state.logical_replica_count.shape[1])
            global_expert_load = None
        else:
            num_local_physical_experts = torch.tensor([num_local_experts],
                                                      dtype=torch.int32,
                                                      device="cpu")
            torch.distributed.broadcast(num_local_physical_experts,
                                        group=get_ep_group().cpu_group,
                                        group_src=0)
            num_local_physical_experts = num_local_physical_experts.item()
            new_physical_experts = num_local_physical_experts * new_ep_size
            assert self.model_runner.eplb_state is not None
            global_expert_load = self.model_runner.eplb_state.rearrange(
                self.model_runner.model, execute_shuffle=False)
            parallel_config.num_redundant_experts = (
                new_physical_experts - global_expert_load.shape[1])
        prepare_communication_buffer_for_model(self.model_runner.model)
        self.model_runner.model.update_physical_experts_metadata(
            num_physical_experts=new_physical_experts,
            num_local_physical_experts=num_local_physical_experts)
        return global_expert_load

    def reinitialize_distributed(
            self, reconfig_request: ReconfigureDistributedRequest) -> None:
        """
        根据重新配置请求重新初始化分布式环境。

        此方法处理分布式并行配置的更新，包括专家并行组的缩放。
        它会在规模缩小前重新分配专家，并在规模扩大后重新分配专家。

        Args:
            reconfig_request: 包含新的分布式配置信息的请求对象。
        """
        from vllm.config import set_current_vllm_config
        from vllm.distributed.parallel_state import (
            cleanup_dist_env_and_memory, get_ep_group)

        old_ep_size = get_ep_group().world_size
        old_ep_rank = get_ep_group().rank
        new_ep_size = reconfig_request.new_data_parallel_size * get_tp_group(
        ).world_size * get_pp_group().world_size
        if new_ep_size < old_ep_size:
            self._eplb_before_scale_down(old_ep_size, new_ep_size)

        cleanup_dist_env_and_memory()

        if reconfig_request.new_data_parallel_rank == \
        ReconfigureRankType.SHUTDOWN_CURRENT_RANK:
            assert old_ep_rank >= new_ep_size
            # shutdown
            return

        self._reconfigure_parallel_config(reconfig_request)

        with set_current_vllm_config(self.vllm_config):
            init_worker_distributed_environment(self.vllm_config, self.rank,
                                                self.distributed_init_method,
                                                self.local_rank)

        global_expert_load = self._reconfigure_moe(old_ep_size, new_ep_size)

        if new_ep_size > old_ep_size:
            assert global_expert_load is not None
            self._eplb_after_scale_up(old_ep_size, new_ep_size,
                                      global_expert_load)

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        """保存模型的分布式（分片）状态。

        此方法用于将模型的当前状态保存到指定路径，支持分片保存和大小限制。

        Args:
            path: 保存模型的路径。
            pattern: 可选的模式，用于过滤要保存的状态。
            max_size: 可选的最大文件大小，用于限制分片文件的大小。
        """
        from vllm.model_executor.model_loader import ShardedStateLoader
        ShardedStateLoader.save_model(
            self.model_runner.model,
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def save_tensorized_model(
        self,
        tensorizer_config: "TensorizerConfig",
    ) -> None:
        """保存张量化模型。

        此方法用于将模型保存为张量化格式，以便后续加载和使用。

        Args:
            tensorizer_config: 张量化配置对象。
        """
        self.model_runner.save_tensorized_model(
            tensorizer_config=tensorizer_config, )


def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
    backend: str = "nccl",
) -> None:
    """初始化Worker的分布式环境。

    此函数设置自定义all-reduce，初始化分布式通信组，
    并确保模型并行和KV传输已正确初始化。

    Args:
        vllm_config: vLLM的配置对象。
        rank: 当前Worker在整个分布式环境中的全局排名。
        distributed_init_method: 分布式初始化方法。
        local_rank: 当前Worker在本地（单个节点内）的GPU排名。
        backend: 分布式通信的后端，默认为“nccl”。
    """
    parallel_config = vllm_config.parallel_config
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_distributed_environment(parallel_config.world_size, rank,
                                 distributed_init_method, local_rank, backend)

    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)

    ensure_kv_transfer_initialized(vllm_config)


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:  # noqa: SIM102
        if not current_platform.has_device_capability(80):
            capability = current_platform.get_device_capability()
            gpu_name = current_platform.get_device_name()

            if capability is None:
                compute_str = "does not have a compute capability"
            else:
                version_str = capability.as_version_str()
                compute_str = f"has compute capability {version_str}"

            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU {compute_str}. "
                "You can use float16 instead by explicitly setting the "
                "`dtype` flag in CLI, for example: --dtype=half.")
