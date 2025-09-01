# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ============================================================================
# vLLM V1 引擎核心模块
# 这个文件是 vLLM V1 引擎的核心实现，包含了引擎的主要逻辑：
# 1. EngineCore - 引擎的内部循环和核心逻辑
# 2. EngineCoreProc - 在后台进程中运行 EngineCore 的 ZMQ 包装器
# 3. DPEngineCoreProc - 支持数据并行的引擎核心进程
# 4. DPEngineCoreActor - 在 Ray 环境中运行的数据并行引擎核心
# ============================================================================

import os
import queue
import signal
import threading
import time
from collections import deque
from collections.abc import Generator
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from inspect import isclass, signature
from logging import DEBUG
from typing import Any, Callable, Optional, TypeVar, Union

import msgspec
import zmq

from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.logger import init_logger
from vllm.logging_utils.dump_input import dump_engine_exception
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tasks import POOLING_TASKS, SupportedTask
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value)
from vllm.utils import (decorate_logs, get_hash_fn_by_name, make_zmq_socket,
                        resolve_obj_by_qualname, set_process_title)
from vllm.v1.core.kv_cache_utils import (BlockHash, get_kv_cache_config,
                                         get_request_block_hasher,
                                         init_none_hash,
                                         unify_kv_cache_configs)
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler as V1Scheduler
from vllm.v1.engine import (EngineCoreOutputs, EngineCoreRequest,
                            EngineCoreRequestType,
                            ReconfigureDistributedRequest, ReconfigureRankType,
                            UtilityOutput, UtilityResult)
from vllm.v1.engine.mm_input_cache import MultiModalInputCacheServer
from vllm.v1.engine.utils import EngineHandshakeMetadata, EngineZmqAddresses
from vllm.v1.executor.abstract import Executor
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.structured_output import StructuredOutputManager
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

# 轮询超时时间（秒）- 用于 ZMQ 套接字轮询
POLLING_TIMEOUT_S = 2.5
# 握手超时时间（分钟）- 引擎与前端进程握手的最大等待时间
HANDSHAKE_TIMEOUT_MINS = 5

# 泛型类型变量，用于 collective_rpc 方法的返回类型
_R = TypeVar('_R')  # Return type for collective_rpc


class EngineCore:
    """
    vLLM 引擎的内部循环核心类
    
    这是 vLLM V1 引擎的核心组件，负责：
    1. 初始化模型执行器、调度器、KV 缓存等组件
    2. 处理请求的添加、中止等操作
    3. 执行模型推理的核心循环（调度 -> 执行 -> 输出）
    4. 管理批处理队列以支持流水线并行
    5. 处理多模态输入缓存和结构化输出
    """

    def __init__(self,
                 vllm_config: VllmConfig,        # vLLM 配置对象，包含所有配置信息
                 executor_class: type[Executor], # 执行器类，负责实际的模型推理
                 log_stats: bool,                # 是否记录统计信息
                 executor_fail_callback: Optional[Callable] = None):  # 执行器失败时的回调函数

        # 在引擎/调度器级别也需要加载插件
        # 插件系统允许用户扩展 vLLM 的功能
        from vllm.plugins import load_general_plugins
        load_general_plugins()

        self.vllm_config = vllm_config
        logger.info("正在初始化 V1 LLM 引擎 (v%s)，配置: %s",
                    VLLM_VERSION, vllm_config)

        self.log_stats = log_stats  # 保存是否记录统计信息的标志

        # 设置模型执行器
        # 执行器负责实际的模型推理，包括前向传播、张量操作等
        self.model_executor = executor_class(vllm_config)
        if executor_fail_callback is not None:
            # 注册执行器失败时的回调函数，用于错误处理
            self.model_executor.register_failure_callback(
                executor_fail_callback)

        # 可用于 KV 缓存的 GPU 内存大小（字节），-1 表示尚未初始化
        self.available_gpu_memory_for_kv_cache = -1

        # 设置 KV 缓存并在性能分析后更新缓存配置
        # KV 缓存用于存储注意力机制中的键值对，是 Transformer 模型推理的关键组件
        num_gpu_blocks, num_cpu_blocks, kv_cache_config = \
            self._initialize_kv_caches(vllm_config)

        # 更新配置中的缓存块数量
        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks  # GPU 上的缓存块数
        vllm_config.cache_config.num_cpu_blocks = num_cpu_blocks  # CPU 上的缓存块数
        # 通过集体 RPC 调用在所有工作进程中初始化缓存
        self.collective_rpc("initialize_cache",
                            args=(num_gpu_blocks, num_cpu_blocks))

        # 结构化输出管理器
        # 负责处理结构化输出（如 JSON schema 约束的输出）
        self.structured_output_manager = StructuredOutputManager(vllm_config)

        # 设置调度器
        # 调度器负责决定哪些请求应该被处理，如何分配资源等
        if isinstance(vllm_config.scheduler_config.scheduler_cls, str):
            # 如果调度器类以字符串形式提供，则通过限定名解析获取实际的类
            Scheduler = resolve_obj_by_qualname(
                vllm_config.scheduler_config.scheduler_cls)
        else:
            # 直接使用提供的调度器类
            Scheduler = vllm_config.scheduler_config.scheduler_cls

        # 这个警告可以在 V1 调度器接口最终确定后移除
        # 届时我们可以维持对实现该接口的调度器类的支持
        if Scheduler is not V1Scheduler:
            logger.warning(
                "使用配置的 V1 调度器类 %s。"
                "此调度器接口不是公共接口，"
                "兼容性可能无法保证。",
                vllm_config.scheduler_config.scheduler_cls)

        if len(kv_cache_config.kv_cache_groups) == 0:
            # 没有 KV 缓存的编码器模型不支持分块预填充
            # 但是 SSM（状态空间模型）模型呢？
            logger.info("为没有 KV 缓存的模型禁用分块预填充")
            vllm_config.scheduler_config.chunked_prefill_enabled = False

        # 创建调度器实例
        self.scheduler: SchedulerInterface = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=self.structured_output_manager,
            # 只有在数据并行大小 > 1 时才包含完成集合，用于跨进程同步
            include_finished_set=vllm_config.parallel_config.data_parallel_size > 1,
            log_stats=self.log_stats,
        )
        # 检查是否启用推测解码（speculative decoding）
        self.use_spec_decode = vllm_config.speculative_config is not None

        # 多模态输入缓存服务器
        # 用于缓存和管理多模态输入（如图像、音频等），避免重复处理
        self.mm_input_cache_server = MultiModalInputCacheServer(
            vllm_config.model_config, MULTIMODAL_REGISTRY)

        # 为流水线并行设置批处理队列
        # 已调度批次的批处理队列。这使我们能够异步调度和执行批次，
        # 这是流水线并行消除流水线气泡所必需的。
        self.batch_queue_size = self.model_executor.max_concurrent_batches
        # 批处理队列：存储 (Future[ModelRunnerOutput], SchedulerOutput) 元组
        self.batch_queue: Optional[queue.Queue[tuple[Future[ModelRunnerOutput],
                                                     SchedulerOutput]]] = None
        if self.batch_queue_size > 1:
            logger.info("批处理队列已启用，大小为 %d",
                        self.batch_queue_size)
            self.batch_queue = queue.Queue(self.batch_queue_size)

        # 请求块哈希器：用于前缀缓存的哈希函数
        self.request_block_hasher: Optional[Callable[[Request],
                                                     list[BlockHash]]] = None
        # 如果启用了前缀缓存或存在 KV 连接器，则初始化块哈希器
        if (self.vllm_config.cache_config.enable_prefix_caching
                or self.scheduler.get_kv_connector() is not None):

            block_size = vllm_config.cache_config.block_size  # 缓存块大小
            # 根据配置的哈希算法名称获取哈希函数
            caching_hash_fn = get_hash_fn_by_name(
                vllm_config.cache_config.prefix_caching_hash_algo)
            # 初始化空哈希值
            init_none_hash(caching_hash_fn)

            # 创建请求块哈希器，用于计算请求的块哈希值
            self.request_block_hasher = get_request_block_hasher(
                block_size, caching_hash_fn)

    def _initialize_kv_caches(
            self, vllm_config: VllmConfig) -> tuple[int, int, KVCacheConfig]:
        """
        初始化 KV 缓存系统
        
        这个方法负责：
        1. 获取模型所需的 KV 缓存规格
        2. 分析可用的 GPU 内存
        3. 计算可以分配多少个缓存块
        4. 统一所有工作进程的缓存配置
        5. 初始化模型执行器的缓存系统
        
        返回：
            tuple[int, int, KVCacheConfig]: (GPU块数, CPU块数, KV缓存配置)
        """
        start = time.time()

        # 获取模型所需的所有 KV 缓存规格
        # 不同的模型层可能需要不同的缓存配置
        kv_cache_specs = self.model_executor.get_kv_cache_specs()

        # 检查模型是否需要 KV 缓存（某些编码器模型不需要）
        has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)
        if has_kv_cache:
            # 检查是否在弹性端点扩容启动模式下
            if os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH") == "1":
                # 在扩容模式下，从数据并行组同步 KV 缓存内存大小
                dp_group = getattr(self, "dp_group", None)
                assert dp_group is not None
                self.available_gpu_memory_for_kv_cache = \
                    ParallelConfig.sync_kv_cache_memory_size(dp_group, -1)
                # 为所有工作进程使用相同的可用内存大小
                available_gpu_memory = [
                    self.available_gpu_memory_for_kv_cache
                ] * len(kv_cache_specs)
            else:
                # 分析模型的峰值内存使用情况，以确定有多少内存可以分配给 KV 缓存
                # 这个过程通过运行模型的前向传播来测量内存使用情况
                available_gpu_memory = (
                    self.model_executor.determine_available_memory())
                self.available_gpu_memory_for_kv_cache = \
                    available_gpu_memory[0]
        else:
            # 无注意力机制的模型不需要 KV 缓存内存
            available_gpu_memory = [0] * len(kv_cache_specs)

        assert len(kv_cache_specs) == len(available_gpu_memory)
        # 计算 KV 缓存张量大小
        # 为每个工作进程生成 KV 缓存配置
        kv_cache_configs = [
            get_kv_cache_config(vllm_config, kv_cache_spec_one_worker,
                                available_gpu_memory_one_worker)
            for kv_cache_spec_one_worker, available_gpu_memory_one_worker in
            zip(kv_cache_specs, available_gpu_memory)
        ]

        # 由于我们使用共享的集中式控制器，我们需要 `kv_cache_config` 在所有工作进程中保持一致
        # 以确保所有内存操作都可以应用于所有工作进程
        unify_kv_cache_configs(kv_cache_configs)

        # 所有工作进程的 kv_cache_config 都相同（除了层名称），因此使用任意一个来初始化调度器
        assert all([
            cfg.num_blocks == kv_cache_configs[0].num_blocks
            for cfg in kv_cache_configs
        ])
        num_gpu_blocks = kv_cache_configs[0].num_blocks  # GPU 上的缓存块数量
        num_cpu_blocks = 0  # CPU 缓存块数量（V1 引擎中暂时设为 0）
        scheduler_kv_cache_config = kv_cache_configs[0]  # 调度器使用的缓存配置

        # 初始化 KV 缓存并预热执行
        # 这会在所有工作进程中创建实际的缓存张量并进行预热运行
        self.model_executor.initialize_from_config(kv_cache_configs)

        elapsed = time.time() - start
        logger.info(("初始化引擎（性能分析、创建 KV 缓存、预热模型）耗时 %.2f 秒"), elapsed)
        return num_gpu_blocks, num_cpu_blocks, scheduler_kv_cache_config

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        """
        获取模型支持的任务类型
        
        返回：
            tuple[SupportedTask, ...]: 支持的任务类型元组，如生成、分类、嵌入等
        """
        return self.model_executor.supported_tasks

    def add_request(self, request: Request, request_wave: int = 0):
        """
        将请求添加到调度器中
        
        这个方法执行以下操作：
        1. 验证请求 ID 的类型
        2. 检查池化参数的有效性（如果有）
        3. 处理 KV 传输参数
        4. 将请求添加到调度器
        
        参数：
            request (Request): 要添加的请求对象
            request_wave (int): 在数据并行情况下，指示此请求预期属于哪一波请求
        """
        # 验证请求 ID 的类型
        # 请求 ID 必须是字符串类型，用于唯一标识每个请求
        if not isinstance(request.request_id, str):
            raise TypeError(
                f"请求 ID 必须是字符串类型，得到 {type(request.request_id)}")

        # 如果请求包含池化参数，验证任务类型是否受支持
        if pooling_params := request.pooling_params:
            # 获取模型支持的池化任务类型
            supported_pooling_tasks = [
                task for task in self.get_supported_tasks()
                if task in POOLING_TASKS
            ]

            # 检查请求的池化任务是否在支持的任务列表中
            if pooling_params.task not in supported_pooling_tasks:
                raise ValueError(f"不支持的任务: {pooling_params.task!r} "
                                 f"支持的任务: {supported_pooling_tasks}")

        # 检查 KV 传输参数的有效性
        if request.kv_transfer_params is not None and (
                not self.scheduler.get_kv_connector()):
            logger.warning("收到 kv_transfer_params，但未找到 KVConnector。"
                           "为此请求禁用 KV 传输。")

        # 将请求添加到调度器中进行处理
        self.scheduler.add_request(request)

    def abort_requests(self, request_ids: list[str]):
        """
        从调度器中中止指定的请求
        
        参数：
            request_ids (list[str]): 要中止的请求 ID 列表
        """

        # TODO: 调度器实际上不需要知道具体的完成原因
        # 待定是否传播该信息（即客户端中止 vs 满足停止条件）
        self.scheduler.finish_requests(request_ids,
                                       RequestStatus.FINISHED_ABORTED)

    def execute_model_with_error_logging(
        self,
        model_fn: Callable[[SchedulerOutput], ModelRunnerOutput],
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput:
        """
        执行模型并在失败时记录详细信息
        
        这个方法包装了模型执行调用，并在发生异常时记录详细的调试信息。
        这对于排查模型执行问题非常有用。
        
        参数：
            model_fn: 模型执行函数，接收 SchedulerOutput 并返回 ModelRunnerOutput
            scheduler_output: 调度器输出，包含要执行的批次信息
            
        返回：
            ModelRunnerOutput: 模型执行结果
        """
        try:
            return model_fn(scheduler_output)
        except Exception as err:
            # 我们不想在这里捕获 BaseException，因为我们只对
            # 由 execute_model 本身的错误引起的异常时转储信息感兴趣

            # 注意：这个方法是无异常的
            dump_engine_exception(self.vllm_config, scheduler_output,
                                  self.scheduler.make_stats())
            raise err

    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """
        执行一次引擎步骤：调度、执行和生成输出
        
        这是引擎的核心循环，包含三个主要步骤：
        1. 调度：决定哪些请求应该被处理
        2. 执行：运行模型推理
        3. 输出：更新调度器状态并生成输出
        
        返回：
            tuple[dict[int, EngineCoreOutputs], bool]: 
                - 第一个元素是输出字典，键为客户端索引，值为引擎输出
                - 第二个元素是标志，指示模型是否被执行
        """

        # 检查调度器中是否还有剩余请求 - 未完成的或已完成但尚未从批次中移除的
        if not self.scheduler.has_requests():
            return {}, False  # 没有请求时返回空结果
        
        # 第一步：调度 - 决定哪些请求应该被处理
        scheduler_output = self.scheduler.schedule()
        
        # 第二步：执行 - 运行模型推理
        model_output = self.execute_model_with_error_logging(
            self.model_executor.execute_model,  # type: ignore
            scheduler_output)
        
        # 第三步：更新 - 根据模型输出更新调度器状态并生成输出
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output)  # type: ignore

        return (engine_core_outputs,
                scheduler_output.total_num_scheduled_tokens > 0)  # 返回输出和模型执行标志

    def post_step(self, model_executed: bool) -> None:
        """
        步骤后处理，主要用于推测解码
        
        在模型执行后调用，处理推测解码相关的后续操作。
        推测解码是一种加速技术，通过并行生成多个候选令牌来提高生成速度。
        
        参数：
            model_executed (bool): 模型是否被执行的标志
        """
        if self.use_spec_decode and model_executed:
            # 获取草稿令牌 ID
            # 在推测解码中，草稿模型生成多个候选令牌，然后由主模型验证
            draft_token_ids = self.model_executor.take_draft_token_ids()
            if draft_token_ids is not None:
                # 更新调度器中的草稿令牌 ID
                self.scheduler.update_draft_token_ids(draft_token_ids)

    def step_with_batch_queue(
            self) -> tuple[Optional[dict[int, EngineCoreOutputs]], bool]:
        """
        使用批处理队列调度和执行批次
        
        注意：如果此步骤中没有输出，则返回 None。
        
        执行流程如下：
        1. 如果批处理队列未满，尝试调度新批次。
           如果调度了新批次，直接返回空的引擎核心输出。
           换句话说，填充批处理队列比获取模型输出具有更高的优先级。
        2. 如果没有新的已调度批次，意味着批处理队列已满或没有其他请求可以调度，
           我们阻塞直到作业队列中的第一个批次完成。
        3. 根据输出更新调度器。
        
        返回：
            tuple[Optional[dict[int, EngineCoreOutputs]], bool]: 
                - 第一个元素是输出字典（可能为 None）
                - 第二个元素是标志，指示是否调度了新批次
        """
        assert self.batch_queue is not None

        engine_core_outputs = None
        scheduler_output = None
        # 如果批处理队列未满，尝试调度新批次，但如果所有请求都已调度，
        # 调度器可能返回空批次。
        # 注意这不是阻塞的。
        if not self.batch_queue.full():
            scheduler_output = self.scheduler.schedule()
            if scheduler_output.total_num_scheduled_tokens > 0:
                # 异步执行模型，返回 Future 对象
                future = self.model_executor.execute_model(scheduler_output)
                # 将 Future 和调度器输出放入批处理队列
                self.batch_queue.put_nowait(
                    (future, scheduler_output))  # type: ignore

        # 检查是否成功调度了新批次
        scheduled_batch = (scheduler_output is not None
                           and scheduler_output.total_num_scheduled_tokens > 0)

        # 如果没有更多请求可以调度且作业队列不为空，
        # 阻塞直到作业队列中的第一个批次完成。
        # TODO(comaniac): 理想情况下，我们应该在调度新批次之前稥视作业队列中的第一个批次
        # 以检查它是否已完成，但稥视队列中的第一个元素不是线程安全的，
        # 所以我们需要做更多工作。
        if not scheduled_batch and not self.batch_queue.empty():
            # 从队列中获取第一个作业
            future, scheduler_output = self.batch_queue.get_nowait()

            # 阻塞直到第一个结果可用
            model_output = self.execute_model_with_error_logging(
                lambda _: future.result(), scheduler_output)

            # 标记任务完成
            self.batch_queue.task_done()
            # 根据模型输出更新调度器并生成引擎输出
            engine_core_outputs = (self.scheduler.update_from_output(
                scheduler_output, model_output))

        return engine_core_outputs, scheduled_batch

    def shutdown(self):
        """
        关闭引擎核心，清理所有资源
        
        这个方法会按顺序关闭所有组件：
        1. 结构化输出管理器
        2. 模型执行器
        3. 调度器
        """
        # 清理结构化输出管理器的后端
        self.structured_output_manager.clear_backend()
        # 关闭模型执行器
        if self.model_executor:
            self.model_executor.shutdown()
        # 关闭调度器
        if self.scheduler:
            self.scheduler.shutdown()

    def profile(self, is_start: bool = True):
        """
        启动或停止性能分析
        
        参数：
            is_start (bool): True 表示开始分析，False 表示停止分析
        """
        self.model_executor.profile(is_start)

    def reset_mm_cache(self):
        """
        重置多模态输入缓存
        
        注意：由于这主要用于调试，我们不尝试重新同步内部缓存
        （P0 处理器、P0 镜像、P1 镜像）
        """
        if self.scheduler.has_unfinished_requests():
            logger.warning("在请求正在进行时重置多模态缓存"
                           "可能导致内部缓存去同步。")

        # 重置多模态输入缓存服务器
        self.mm_input_cache_server.reset()

    def reset_prefix_cache(self):
        """
        重置前缀缓存
        
        前缀缓存用于存储常见的请求前缀，以加速具有相同前缀的请求处理。
        """
        self.scheduler.reset_prefix_cache()

    def sleep(self, level: int = 1):
        """
        让模型执行器进入睡眠模式
        
        睡眠模式用于节省资源，在没有请求时减少 GPU 使用。
        
        参数：
            level (int): 睡眠级别，数值越高节省资源越多
        """
        self.model_executor.sleep(level)

    def wake_up(self, tags: Optional[list[str]] = None):
        """
        唤醒模型执行器
        
        从睡眠模式中唤醒模型执行器，使其能够处理新请求。
        
        参数：
            tags (Optional[list[str]]): 可选的标签列表，用于指定唤醒特定组件
        """
        self.model_executor.wake_up(tags)

    def is_sleeping(self) -> bool:
        """
        检查模型执行器是否在睡眠模式
        
        返回：
            bool: True 表示正在睡眠，False 表示正在活动
        """
        return self.model_executor.is_sleeping

    def execute_dummy_batch(self):
        """
        执行虚拟批次
        
        在数据并行情况下，当没有实际请求需要处理但需要保持同步时，
        执行一个空的模型前向传播以保持所有进程的同步。
        """
        self.model_executor.collective_rpc("execute_dummy_batch")

    def add_lora(self, lora_request: LoRARequest) -> bool:
        """
        添加 LoRA 适配器
        
        LoRA (Low-Rank Adaptation) 是一种参数高效的微调方法，
        允许在不修改原始模型参数的情况下适配新任务。
        
        参数：
            lora_request (LoRARequest): LoRA 请求对象，包含适配器的配置信息
            
        返回：
            bool: 是否成功添加 LoRA 适配器
        """
        return self.model_executor.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        """
        移除 LoRA 适配器
        
        从模型中移除指定的 LoRA 适配器，释放相关资源。
        
        参数：
            lora_id (int): 要移除的 LoRA 适配器 ID
            
        返回：
            bool: 是否成功移除 LoRA 适配器
        """
        return self.model_executor.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        """
        列出所有已加载的 LoRA 适配器
        
        返回：
            set[int]: 已加载的 LoRA 适配器 ID 集合
        """
        return self.model_executor.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        """
        固定 LoRA 适配器
        
        将指定的 LoRA 适配器固定在内存中，防止其被自动卸载。
        这对于频繁使用的适配器很有用。
        
        参数：
            lora_id (int): 要固定的 LoRA 适配器 ID
            
        返回：
            bool: 是否成功固定 LoRA 适配器
        """
        return self.model_executor.pin_lora(lora_id)

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        """
        保存分片模型状态
        
        将模型的状态字典保存为多个分片文件，用于大型模型的检查点保存。
        
        参数：
            path (str): 保存路径
            pattern (Optional[str]): 文件名模式，用于生成分片文件名
            max_size (Optional[int]): 每个分片文件的最大大小（字节）
        """
        self.model_executor.save_sharded_state(path=path,
                                               pattern=pattern,
                                               max_size=max_size)

    def collective_rpc(self,
                       method: Union[str, Callable[..., _R]],
                       timeout: Optional[float] = None,
                       args: tuple = (),
                       kwargs: Optional[dict[str, Any]] = None) -> list[_R]:
        """
        集体 RPC 调用
        
        在所有工作进程中执行相同的方法调用，用于分布式模型的同步操作。
        
        参数：
            method (Union[str, Callable]): 要调用的方法名或方法对象
            timeout (Optional[float]): 超时时间（秒）
            args (tuple): 位置参数
            kwargs (Optional[dict]): 关键字参数
            
        返回：
            list[_R]: 所有工作进程返回结果的列表
        """
        return self.model_executor.collective_rpc(method, timeout, args,
                                                  kwargs)

    def save_tensorized_model(
        self,
        tensorizer_config,
    ) -> None:
        """
        保存张量化模型
        
        使用 Tensorizer 库将模型保存为优化格式，以加速模型加载。
        
        参数：
            tensorizer_config: Tensorizer 配置对象
        """
        self.model_executor.save_tensorized_model(
            tensorizer_config=tensorizer_config, )

    def preprocess_add_request(
            self, request: EngineCoreRequest) -> tuple[Request, int]:
        """
        预处理请求
        
        这个函数可以直接在输入处理线程中使用，允许请求初始化与模型前向传播并行运行。
        
        这个方法执行以下操作：
        1. 处理多模态输入的哈希和缓存
        2. 将 EngineCoreRequest 转换为 Request 对象
        3. 初始化结构化输出语法（如果需要）
        
        参数：
            request (EngineCoreRequest): 原始的引擎核心请求
            
        返回：
            tuple[Request, int]: 处理后的请求对象和请求波次
        """
        # 处理多模态输入的哈希和缓存
        if request.mm_hashes is not None:
            assert request.mm_kwargs is not None

            # 线程安全注意事项：没有竞态条件。
            # `mm_input_cache_server` 在 LLMEngine 初始化结束时被重置，
            # 之后只会在输入处理线程中访问。
            request.mm_kwargs = self.mm_input_cache_server.get_and_update(
                request.mm_kwargs, request.mm_hashes)

        # 将 EngineCoreRequest 转换为 Request 对象
        req = Request.from_engine_core_request(request,
                                               self.request_block_hasher)
        # 如果需要结构化输出，初始化语法编译器
        if req.use_structured_output:
            # 线程安全注意事项：没有竞态条件。
            # `grammar_init` 只在输入处理线程中调用。对于
            # `structured_output_manager`，每个请求都是独立的，语法编译是异步的。
            # 调度器在调度请求之前总是检查语法编译状态。
            self.structured_output_manager.grammar_init(req)
        return req, request.current_wave  # 返回处理后的请求和波次号


class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""

    ENGINE_CORE_DEAD = b'ENGINE_CORE_DEAD'

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: Optional[str] = None,
        engine_index: int = 0,
    ):
        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
        self.output_queue = queue.Queue[Union[tuple[int, EngineCoreOutputs],
                                              bytes]]()
        executor_fail_callback = lambda: self.input_queue.put_nowait(
            (EngineCoreRequestType.EXECUTOR_FAILED, b''))

        self.engine_index = engine_index
        identity = self.engine_index.to_bytes(length=2, byteorder="little")
        self.engines_running = False

        with self._perform_handshakes(handshake_address, identity,
                                      local_client, vllm_config,
                                      client_handshake_address) as addresses:
            self.client_count = len(addresses.outputs)

            # Set up data parallel environment.
            self.has_coordinator = addresses.coordinator_output is not None
            self.frontend_stats_publish_address = (
                addresses.frontend_stats_publish_address)
            logger.debug("Has DP Coordinator: %s, stats publish address: %s",
                         self.has_coordinator,
                         self.frontend_stats_publish_address)
            # Only publish request queue stats to coordinator for "internal"
            # and "hybrid" LB modes .
            self.publish_dp_lb_stats = (
                self.has_coordinator
                and not vllm_config.parallel_config.data_parallel_external_lb)

            self._init_data_parallel(vllm_config)

            super().__init__(vllm_config, executor_class, log_stats,
                             executor_fail_callback)

            # Background Threads and Queues for IO. These enable us to
            # overlap ZMQ socket IO with GPU since they release the GIL,
            # and to overlap some serialization/deserialization with the
            # model forward pass.
            # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
            ready_event = threading.Event()
            input_thread = threading.Thread(target=self.process_input_sockets,
                                            args=(addresses.inputs,
                                                  addresses.coordinator_input,
                                                  identity, ready_event),
                                            daemon=True)
            input_thread.start()

            self.output_thread = threading.Thread(
                target=self.process_output_sockets,
                args=(addresses.outputs, addresses.coordinator_output,
                      self.engine_index),
                daemon=True)
            self.output_thread.start()

            # Don't complete handshake until DP coordinator ready message is
            # received.
            while not ready_event.wait(timeout=10):
                if not input_thread.is_alive():
                    raise RuntimeError(
                        "Input socket thread died during startup")
                assert addresses.coordinator_input is not None
                logger.info("Waiting for READY message from DP Coordinator...")

        self.step_fn = (self.step if self.batch_queue is None else
                        self.step_with_batch_queue)

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: Optional[str],
    ) -> Generator[EngineZmqAddresses, None, None]:
        """
        Perform startup handshakes.

        For DP=1 or offline mode, this is with the colocated front-end process.

        For DP>1 with internal load-balancing this is with the shared front-end
        process which may reside on a different node.

        For DP>1 with external or hybrid load-balancing, two handshakes are
        performed:
            - With the rank 0 front-end process which retrieves the
              DP Coordinator ZMQ addresses and DP process group address.
            - With the colocated front-end process which retrieves the
              client input/output socket addresses.
        with the exception of the rank 0 and colocated engines themselves which
        don't require the second handshake.

        Here, "front-end" process can mean the process containing the engine
        core client (which is the API server process in the case the API
        server is not scaled out), OR the launcher process running the
        run_multi_api_server() function in serve.py.
        """
        input_ctx = zmq.Context()
        is_local = local_client and client_handshake_address is None
        headless = not local_client
        handshake = self._perform_handshake(input_ctx, handshake_address,
                                            identity, is_local, headless,
                                            vllm_config,
                                            vllm_config.parallel_config)
        if client_handshake_address is None:
            with handshake as addresses:
                yield addresses
        else:
            assert local_client
            local_handshake = self._perform_handshake(
                input_ctx, client_handshake_address, identity, True, False,
                vllm_config)
            with handshake as addresses, local_handshake as client_addresses:
                addresses.inputs = client_addresses.inputs
                addresses.outputs = client_addresses.outputs
                yield addresses

        # Update config which may have changed from the handshake
        vllm_config.__post_init__()

    @contextmanager
    def _perform_handshake(
        self,
        ctx: zmq.Context,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        headless: bool,
        vllm_config: VllmConfig,
        parallel_config_to_update: Optional[ParallelConfig] = None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        with make_zmq_socket(ctx,
                             handshake_address,
                             zmq.DEALER,
                             identity=identity,
                             linger=5000,
                             bind=False) as handshake_socket:
            # Register engine with front-end.
            addresses = self.startup_handshake(handshake_socket, local_client,
                                               headless,
                                               parallel_config_to_update)
            yield addresses

            # Send ready message.
            num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks
            # We pass back the coordinator stats update address here for the
            # external LB case for our colocated front-end to use (coordinator
            # only runs with rank 0).
            dp_stats_address = self.frontend_stats_publish_address
            handshake_socket.send(
                msgspec.msgpack.encode({
                    "status": "READY",
                    "local": local_client,
                    "headless": headless,
                    "num_gpu_blocks": num_gpu_blocks,
                    "dp_stats_address": dp_stats_address,
                }))

    @staticmethod
    def startup_handshake(
        handshake_socket: zmq.Socket,
        local_client: bool,
        headless: bool,
        parallel_config: Optional[ParallelConfig] = None,
    ) -> EngineZmqAddresses:

        # Send registration message.
        handshake_socket.send(
            msgspec.msgpack.encode({
                "status": "HELLO",
                "local": local_client,
                "headless": headless,
            }))

        # Receive initialization message.
        logger.info("Waiting for init message from front-end.")
        if not handshake_socket.poll(timeout=HANDSHAKE_TIMEOUT_MINS * 60_000):
            raise RuntimeError("Did not receive response from front-end "
                               f"process within {HANDSHAKE_TIMEOUT_MINS} "
                               f"minutes")
        init_bytes = handshake_socket.recv()
        init_message: EngineHandshakeMetadata = msgspec.msgpack.decode(
            init_bytes, type=EngineHandshakeMetadata)
        logger.debug("Received init message: %s", init_message)

        if parallel_config is not None:
            for key, value in init_message.parallel_config.items():
                setattr(parallel_config, key, value)

        return init_message.addresses

    @staticmethod
    def run_engine_core(*args,
                        dp_rank: int = 0,
                        local_dp_rank: int = 0,
                        **kwargs):
        """Launch EngineCore busy loop in background process."""

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the engine_core
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        engine_core: Optional[EngineCoreProc] = None
        try:
            parallel_config: ParallelConfig = kwargs[
                "vllm_config"].parallel_config
            if parallel_config.data_parallel_size > 1 or dp_rank > 0:
                set_process_title("DPEngineCore", str(dp_rank))
                decorate_logs()
                # Set data parallel rank for this engine process.
                parallel_config.data_parallel_rank = dp_rank
                parallel_config.data_parallel_rank_local = local_dp_rank
                engine_core = DPEngineCoreProc(*args, **kwargs)
            else:
                set_process_title("EngineCore")
                decorate_logs()
                engine_core = EngineCoreProc(*args, **kwargs)

            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception as e:
            if engine_core is None:
                logger.exception("EngineCore failed to start.")
            else:
                logger.exception("EngineCore encountered a fatal error.")
                engine_core._send_engine_dead()
            raise e
        finally:
            if engine_core is not None:
                engine_core.shutdown()

    def _init_data_parallel(self, vllm_config: VllmConfig):
        pass

    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # 2) Step the engine core and return the outputs.
            self._process_engine_step()

    def _process_input_queue(self):
        """Exits when an engine step needs to be performed."""

        waited = False
        while not self.engines_running and not self.scheduler.has_requests():
            if logger.isEnabledFor(DEBUG) and self.input_queue.empty():
                logger.debug("EngineCore waiting for work.")
                waited = True
            req = self.input_queue.get()
            self._handle_client_request(*req)

        if waited:
            logger.debug("EngineCore loop active.")

        # Handle any more client requests.
        while not self.input_queue.empty():
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)

    def _process_engine_step(self) -> bool:
        """Called only when there are unfinished local requests."""

        # Step the engine core.
        outputs, model_executed = self.step_fn()
        # Put EngineCoreOutputs into the output queue.
        for output in (outputs.items() if outputs else ()):
            self.output_queue.put_nowait(output)
        # Post-step hook.
        self.post_step(model_executed)

        return model_executed

    def _handle_client_request(self, request_type: EngineCoreRequestType,
                               request: Any) -> None:
        """Dispatch request from client."""

        if request_type == EngineCoreRequestType.ADD:
            req, request_wave = request
            self.add_request(req, request_wave)
        elif request_type == EngineCoreRequestType.ABORT:
            self.abort_requests(request)
        elif request_type == EngineCoreRequestType.UTILITY:
            client_idx, call_id, method_name, args = request
            output = UtilityOutput(call_id)
            try:
                method = getattr(self, method_name)
                result = method(*self._convert_msgspec_args(method, args))
                output.result = UtilityResult(result)
            except BaseException as e:
                logger.exception("Invocation of %s method failed", method_name)
                output.failure_message = (f"Call to {method_name} method"
                                          f" failed: {str(e)}")
            self.output_queue.put_nowait(
                (client_idx, EngineCoreOutputs(utility_output=output)))
        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
            raise RuntimeError("Executor failed.")
        else:
            logger.error("Unrecognized input request type encountered: %s",
                         request_type)

    @staticmethod
    def _convert_msgspec_args(method, args):
        """If a provided arg type doesn't match corresponding target method
         arg type, try converting to msgspec object."""
        if not args:
            return args
        arg_types = signature(method).parameters.values()
        assert len(args) <= len(arg_types)
        return tuple(
            msgspec.convert(v, type=p.annotation) if isclass(p.annotation)
            and issubclass(p.annotation, msgspec.Struct)
            and not isinstance(v, p.annotation) else v
            for v, p in zip(args, arg_types))

    def _send_engine_dead(self):
        """Send EngineDead status to the EngineCoreClient."""

        # Put ENGINE_CORE_DEAD in the queue.
        self.output_queue.put_nowait(EngineCoreProc.ENGINE_CORE_DEAD)

        # Wait until msg sent by the daemon before shutdown.
        self.output_thread.join(timeout=5.0)
        if self.output_thread.is_alive():
            logger.fatal("vLLM shutdown signal from EngineCore failed "
                         "to send. Please report this issue.")

    def process_input_sockets(self, input_addresses: list[str],
                              coord_input_address: Optional[str],
                              identity: bytes, ready_event: threading.Event):
        """Input socket IO thread."""

        # Msgpack serialization decoding.
        add_request_decoder = MsgpackDecoder(EngineCoreRequest)
        generic_decoder = MsgpackDecoder()

        with ExitStack() as stack, zmq.Context() as ctx:
            input_sockets = [
                stack.enter_context(
                    make_zmq_socket(ctx,
                                    input_address,
                                    zmq.DEALER,
                                    identity=identity,
                                    bind=False))
                for input_address in input_addresses
            ]
            if coord_input_address is None:
                coord_socket = None
            else:
                coord_socket = stack.enter_context(
                    make_zmq_socket(ctx,
                                    coord_input_address,
                                    zmq.XSUB,
                                    identity=identity,
                                    bind=False))
                # Send subscription message to coordinator.
                coord_socket.send(b'\x01')

            # Register sockets with poller.
            poller = zmq.Poller()
            for input_socket in input_sockets:
                # Send initial message to each input socket - this is required
                # before the front-end ROUTER socket can send input messages
                # back to us.
                input_socket.send(b'')
                poller.register(input_socket, zmq.POLLIN)

            if coord_socket is not None:
                # Wait for ready message from coordinator.
                assert coord_socket.recv() == b"READY"
                poller.register(coord_socket, zmq.POLLIN)

            ready_event.set()
            del ready_event
            while True:
                for input_socket, _ in poller.poll():
                    # (RequestType, RequestData)
                    type_frame, *data_frames = input_socket.recv_multipart(
                        copy=False)
                    request_type = EngineCoreRequestType(
                        bytes(type_frame.buffer))

                    # Deserialize the request data.
                    if request_type == EngineCoreRequestType.ADD:
                        request = add_request_decoder.decode(data_frames)
                        request = self.preprocess_add_request(request)
                    else:
                        request = generic_decoder.decode(data_frames)

                    # Push to input queue for core busy loop.
                    self.input_queue.put_nowait((request_type, request))

    def process_output_sockets(self, output_paths: list[str],
                               coord_output_path: Optional[str],
                               engine_index: int):
        """Output socket IO thread."""

        # Msgpack serialization encoding.
        encoder = MsgpackEncoder()
        # Send buffers to reuse.
        reuse_buffers: list[bytearray] = []
        # Keep references to outputs and buffers until zmq is finished
        # with them (outputs may contain tensors/np arrays whose
        # backing buffers were extracted for zero-copy send).
        pending = deque[tuple[zmq.MessageTracker, Any, bytearray]]()

        # We must set linger to ensure the ENGINE_CORE_DEAD
        # message is sent prior to closing the socket.
        with ExitStack() as stack, zmq.Context() as ctx:
            sockets = [
                stack.enter_context(
                    make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000))
                for output_path in output_paths
            ]
            coord_socket = stack.enter_context(
                make_zmq_socket(
                    ctx, coord_output_path, zmq.PUSH, bind=False,
                    linger=4000)) if coord_output_path is not None else None
            max_reuse_bufs = len(sockets) + 1

            while True:
                output = self.output_queue.get()
                if output == EngineCoreProc.ENGINE_CORE_DEAD:
                    for socket in sockets:
                        socket.send(output)
                    break
                assert not isinstance(output, bytes)
                client_index, outputs = output
                outputs.engine_index = engine_index

                if client_index == -1:
                    # Don't reuse buffer for coordinator message
                    # which will be very small.
                    assert coord_socket is not None
                    coord_socket.send_multipart(encoder.encode(outputs))
                    continue

                # Reclaim buffers that zmq is finished with.
                while pending and pending[-1][0].done:
                    reuse_buffers.append(pending.pop()[2])

                buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
                buffers = encoder.encode_into(outputs, buffer)
                tracker = sockets[client_index].send_multipart(buffers,
                                                               copy=False,
                                                               track=True)
                if not tracker.done:
                    ref = outputs if len(buffers) > 1 else None
                    pending.appendleft((tracker, ref, buffer))
                elif len(reuse_buffers) < max_reuse_bufs:
                    # Limit the number of buffers to reuse.
                    reuse_buffers.append(buffer)


class DPEngineCoreProc(EngineCoreProc):
    """ZMQ-wrapper for running EngineCore in background process
    in a data parallel context."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: Optional[str] = None,
    ):
        # Counts forward-passes of the model so that we can synchronize
        # finished with DP peers every N steps.
        self.step_counter = 0
        self.current_wave = 0
        self.last_counts = (0, 0)

        # Initialize the engine.
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        super().__init__(vllm_config, local_client, handshake_address,
                         executor_class, log_stats, client_handshake_address,
                         dp_rank)

    def _init_data_parallel(self, vllm_config: VllmConfig):

        # Configure GPUs and stateless process group for data parallel.
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        dp_size = vllm_config.parallel_config.data_parallel_size
        local_dp_rank = vllm_config.parallel_config.data_parallel_rank_local

        assert dp_size > 1
        assert 0 <= local_dp_rank <= dp_rank < dp_size

        if vllm_config.kv_transfer_config is not None:
            # modify the engine_id and append the local_dp_rank to it to ensure
            # that the kv_transfer_config is unique for each DP rank.
            vllm_config.kv_transfer_config.engine_id = (
                f"{vllm_config.kv_transfer_config.engine_id}_dp{local_dp_rank}"
            )
            logger.debug("Setting kv_transfer_config.engine_id to %s",
                         vllm_config.kv_transfer_config.engine_id)

        self.dp_rank = dp_rank
        self.dp_group = vllm_config.parallel_config.stateless_init_dp_group()

    def shutdown(self):
        super().shutdown()
        if dp_group := getattr(self, "dp_group", None):
            stateless_destroy_torch_distributed_process_group(dp_group)

    def add_request(self, request: Request, request_wave: int = 0):
        if self.has_coordinator and request_wave != self.current_wave:
            if request_wave > self.current_wave:
                self.current_wave = request_wave
            elif not self.engines_running:
                # Request received for an already-completed wave, notify
                # front-end that we need to start the next one.
                self.output_queue.put_nowait(
                    (-1, EngineCoreOutputs(start_wave=self.current_wave)))

        super().add_request(request, request_wave)

    def _handle_client_request(self, request_type: EngineCoreRequestType,
                               request: Any) -> None:
        if request_type == EngineCoreRequestType.START_DP_WAVE:
            new_wave, exclude_eng_index = request
            if exclude_eng_index != self.engine_index and (
                    new_wave >= self.current_wave):
                self.current_wave = new_wave
                if not self.engines_running:
                    logger.debug("EngineCore starting idle loop for wave %d.",
                                 new_wave)
                    self.engines_running = True
        else:
            super()._handle_client_request(request_type, request)

    def _maybe_publish_request_counts(self):
        if not self.publish_dp_lb_stats:
            return

        # Publish our request counts (if they've changed).
        counts = self.scheduler.get_request_counts()
        if counts != self.last_counts:
            self.last_counts = counts
            stats = SchedulerStats(*counts,
                                   step_counter=self.step_counter,
                                   current_wave=self.current_wave)
            self.output_queue.put_nowait(
                (-1, EngineCoreOutputs(scheduler_stats=stats)))

    def run_busy_loop(self):
        """Core busy loop of the EngineCore for data parallel case."""

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()

            # 2) Step the engine core.
            executed = self._process_engine_step()
            self._maybe_publish_request_counts()

            local_unfinished_reqs = self.scheduler.has_unfinished_requests()
            if not executed:
                if not local_unfinished_reqs and not self.engines_running:
                    # All engines are idle.
                    continue

                # We are in a running state and so must execute a dummy pass
                # if the model didn't execute any ready requests.
                self.execute_dummy_batch()

            # 3) All-reduce operation to determine global unfinished reqs.
            self.engines_running = self._has_global_unfinished_reqs(
                local_unfinished_reqs)

            if not self.engines_running:
                if self.dp_rank == 0 or not self.has_coordinator:
                    # Notify client that we are pausing the loop.
                    logger.debug("Wave %d finished, pausing engine loop.",
                                 self.current_wave)
                    # In the coordinator case, dp rank 0 sends updates to the
                    # coordinator. Otherwise (offline spmd case), each rank
                    # sends the update to its colocated front-end process.
                    client_index = -1 if self.has_coordinator else 0
                    self.output_queue.put_nowait(
                        (client_index,
                         EngineCoreOutputs(wave_complete=self.current_wave)))
                # Increment wave count and reset step counter.
                self.current_wave += 1
                self.step_counter = 0

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:

        # Optimization - only perform finish-sync all-reduce every 32 steps.
        self.step_counter += 1
        if self.step_counter % 32 != 0:
            return True

        return ParallelConfig.has_unfinished_dp(self.dp_group,
                                                local_unfinished)

    def reinitialize_distributed(
            self, reconfig_request: ReconfigureDistributedRequest) -> None:
        stateless_destroy_torch_distributed_process_group(self.dp_group)
        self.shutdown()

        parallel_config = self.vllm_config.parallel_config
        old_dp_size = parallel_config.data_parallel_size
        parallel_config.data_parallel_size = \
            reconfig_request.new_data_parallel_size
        if reconfig_request.new_data_parallel_rank != -1:
            parallel_config.data_parallel_rank = \
                reconfig_request.new_data_parallel_rank
        # local rank specifies device visibility, it should not be changed
        assert reconfig_request.new_data_parallel_rank_local == \
            ReconfigureRankType.KEEP_CURRENT_RANK
        parallel_config.data_parallel_master_ip = \
            reconfig_request.new_data_parallel_master_ip
        parallel_config.data_parallel_master_port = \
            reconfig_request.new_data_parallel_master_port
        if reconfig_request.new_data_parallel_rank != -2:
            self.dp_rank = parallel_config.data_parallel_rank
            self.dp_group = parallel_config.stateless_init_dp_group()
        reconfig_request.new_data_parallel_master_port = \
            parallel_config.data_parallel_master_port

        self.model_executor.reinitialize_distributed(reconfig_request)
        if reconfig_request.new_data_parallel_size > old_dp_size:
            assert self.available_gpu_memory_for_kv_cache > 0
            # pass available_gpu_memory_for_kv_cache from existing
            # engine-cores to new engine-cores so they can directly
            # use it in _initialize_kv_caches() rather than profiling.
            ParallelConfig.sync_kv_cache_memory_size(
                self.dp_group, self.available_gpu_memory_for_kv_cache)
            # NOTE(yongji): newly joined workers require dummy_run even
            # CUDA graph is not used
            self.model_executor.collective_rpc("compile_or_warm_up_model")
        if reconfig_request.new_data_parallel_rank == \
        ReconfigureRankType.SHUTDOWN_CURRENT_RANK:
            self.shutdown()
            logger.info("DPEngineCoreProc %s shutdown", self.dp_rank)
        else:
            logger.info("Distributed environment reinitialized for DP rank %s",
                        self.dp_rank)


class DPEngineCoreActor(DPEngineCoreProc):
    """
    Ray actor for running EngineCore in a data parallel context
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        self.addresses = addresses
        vllm_config.parallel_config.data_parallel_rank = dp_rank
        vllm_config.parallel_config.data_parallel_rank_local = \
            local_dp_rank

        # Set CUDA_VISIBLE_DEVICES as early as possible in actor life cycle
        # NOTE: in MP we set CUDA_VISIBLE_DEVICES at process creation time,
        # and this cannot be done in the same way for Ray because:
        # 1) Ray manages life cycle of all ray workers (including
        # DPEngineCoreActor)
        # 2) Ray sets CUDA_VISIBLE_DEVICES based on num_gpus configuration
        # To bypass 2, we need to also set
        # RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES, but vLLM workers created
        # thereafter would have CUDA_VISIBLE_DEVICES set, which is sticky:
        # https://github.com/ray-project/ray/blob/e752fc319ddedd9779a0989b6d3613909bad75c9/python/ray/_private/worker.py#L456 # noqa: E501
        # This is problematic because when the vLLM worker (a Ray actor)
        # executes a task, it indexes into the sticky CUDA_VISIBLE_DEVICES
        # rather than directly using the GPU ID, potentially resulting in
        # index out of bounds error. See:
        # https://github.com/ray-project/ray/pull/40461/files#diff-31e8159767361e4bc259b6d9883d9c0d5e5db780fcea4a52ead4ee3ee4a59a78R1860 # noqa: E501
        # and get_accelerator_ids_for_accelerator_resource() in worker.py
        # of ray.
        self._set_cuda_visible_devices(vllm_config, local_dp_rank)

        super().__init__(vllm_config, local_client, "", executor_class,
                         log_stats)

    def _set_cuda_visible_devices(self, vllm_config: VllmConfig,
                                  local_dp_rank: int):
        from vllm.platforms import current_platform
        device_control_env_var = current_platform.device_control_env_var
        world_size = vllm_config.parallel_config.world_size
        # Set CUDA_VISIBLE_DEVICES or equivalent.
        try:
            os.environ[device_control_env_var] = ",".join(
                str(current_platform.device_id_to_physical_device_id(i))
                for i in range(local_dp_rank *
                               world_size, (local_dp_rank + 1) * world_size))
        except IndexError as e:
            raise Exception(
                f"Error setting {device_control_env_var}: "
                f"local range: [{local_dp_rank * world_size}, "
                f"{(local_dp_rank + 1) * world_size}) "
                f"base value: \"{os.getenv(device_control_env_var)}\"") from e

    @contextmanager
    def _perform_handshakes(self, handshake_address: str, identity: bytes,
                            local_client: bool, vllm_config: VllmConfig,
                            client_handshake_address: Optional[str]):
        """
        For Ray, we don't need to actually perform handshake.
        All addresses information is known before the actor creation.
        Therefore, we simply yield these addresses.
        """
        yield self.addresses

    def wait_for_init(self):
        """
        Wait until the engine core is initialized.

        This is just an empty method. When ray.get() on this method
        (or any other method of the actor) returns, it is guaranteed
        that actor creation (i.e., __init__) is complete.
        """
        pass

    def run(self):
        """
        Run the engine core busy loop.
        """
        try:
            self.run_busy_loop()
        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception:
            logger.exception("EngineCore encountered a fatal error.")
            raise
        finally:
            self.shutdown()
