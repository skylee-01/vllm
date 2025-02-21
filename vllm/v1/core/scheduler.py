# SPDX-License-Identifier: Apache-2.0

from collections import deque
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Deque, Dict, Iterable, List, Optional, Set,
                    Tuple, Union)

from vllm.config import CacheConfig, LoRAConfig, ModelConfig, SchedulerConfig
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.encoder_cache_manager import (EncoderCacheManager,
                                                compute_encoder_budget)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

if TYPE_CHECKING:
    from vllm.multimodal import MultiModalKwargs
    from vllm.multimodal.base import PlaceholderRange

logger = init_logger(__name__)


class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.lora_config = lora_config

        # Scheduling constraints. 调度约束。
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len

        num_gpu_blocks = cache_config.num_gpu_blocks
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        # Create the KV cache manager.  # 初始化kv cache管理类。
        self.kv_cache_manager = KVCacheManager(
            block_size=self.cache_config.block_size, # 块大小
            num_gpu_blocks=num_gpu_blocks, # gpu块
            max_model_len=self.max_model_len, # 模型长度
            sliding_window=self.cache_config.sliding_window, # 滑动窗口
            enable_caching=self.cache_config.enable_prefix_caching) # 前缀缓存。
        self.block_size = self.cache_config.block_size

        # req_id -> Request
        self.requests: Dict[str, Request] = {} # 请求id的字典。
        # Priority queues for requests.
        self.waiting: Deque[Request] = deque() # 等待队列
        self.running: List[Request] = [] # 运行队列。

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: Set[str] = set()  # 完成的ids

        # OPTIMIZATION: Cache the CachedRequestData objects to avoid creating
        # them at each scheduling step.
        # Request id -> CachedRequestData
        self._cached_reqs_data: Dict[str, CachedRequestData] = {}  # 缓存请求，避免每次调度都创建。

        # Encoder-related.
        # Calculate encoder cache size if applicable
        # NOTE: For now we use the same budget for both compute and space.
        # This can be changed when we make encoder cache for embedding caching
        # across requests.
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=model_config,
            scheduler_config=scheduler_config,
        )

        # NOTE(woosuk): Here, "encoder" includes the vision encoder (and
        # projector if needed). Currently, we assume that the encoder also
        # has the Transformer architecture (e.g., ViT).
        self.max_num_encoder_input_tokens = encoder_compute_budget
        # NOTE: For the models without encoder (e.g., text-only models),
        # the encoder cache will not be initialized because cache size is 0
        # for these models.
        self.encoder_cache_manager = EncoderCacheManager(
            cache_size=encoder_cache_size)

    def schedule(self) -> "SchedulerOutput":
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and num_tokens,
        # which is equal to len(prompt_token_ids) + len(output_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens. This is general enough to cover chunked prefills,
        # prefix caching, and the "jump decoding" optimization in the future.
        """
        关于调度算法的说明（作者：woosuk）：调度器中不存在“解码阶段”或“预填充阶段”。
        每个请求仅包含  num_computed_tokens  和  num_tokens  ，
        其中  num_tokens  等于  len(prompt_token_ids) + len(output_token_ids)  。
        在每一步中，调度器会尝试将令牌分配给各个请求，以便每个请求的  num_computed_tokens
        能够赶上其  num_tokens  。这种方式足够通用，能够涵盖分块预填充、前缀缓存，
        以及未来可能出现的“跳跃解码”优化。
        """

        scheduled_new_reqs: List[Request] = [] # 新的请求
        scheduled_resumed_reqs: List[Request] = [] # 恢复的请求
        scheduled_running_reqs: List[Request] = [] # 正在运行的请求
        preempted_reqs: List[Request] = [] # 被抢占的请求

        req_to_new_block_ids: Dict[str, List[int]] = {} # 请求到新块的映射
        num_scheduled_tokens: Dict[str, int] = {} # 已调度的令牌数量
        token_budget = self.max_num_scheduled_tokens # 调度预算
        # Encoder-related. # 编码器相关
        scheduled_encoder_inputs: Dict[str, List[int]] = {} # 已调度的输入
        encoder_budget = self.max_num_encoder_input_tokens # 编码器预算

        # First, schedule the RUNNING requests. 首先调度运行中的请求。
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index] # 计算当前query需要的token数量。
            num_new_tokens = request.num_tokens - request.num_computed_tokens
            num_new_tokens = min(num_new_tokens, token_budget)
            assert num_new_tokens > 0

            # Schedule encoder inputs.
            encoder_inputs_to_schedule, num_new_tokens, new_encoder_budget = (
                self._try_schedule_encoder_inputs(request,
                                                  request.num_computed_tokens,
                                                  num_new_tokens,
                                                  encoder_budget))
            if num_new_tokens == 0:
                # The request cannot be scheduled because the encoder budget
                # or the encoder cache is exhausted.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            while True: # 如果kv cache没有空间，则将running队列中最后一个请求转移到waiting队列。
                new_blocks = self.kv_cache_manager.allocate_slots( # 分配新块
                    request, num_new_tokens)
                if new_blocks is None: # 如果没有可用的块，则尝试抢占最不优先的请求
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    preempted_req = self.running.pop() # 获取最不优先的请求
                    self.kv_cache_manager.free(preempted_req) # 释放块
                    preempted_req.status = RequestStatus.PREEMPTED # 设置抢占状态
                    preempted_req.num_computed_tokens = 0 # 设置已计算令牌数量为0

                    self.waiting.appendleft(preempted_req) # 将抢占的请求加入等待队列
                    preempted_reqs.append(preempted_req) # 记录抢占的requ
                    if preempted_req == request: # 如果抢占的请求是当前请求，则跳出循环
                        # No more request to preempt.
                        can_schedule = False
                        break
                else:
                    # The request can be scheduled.
                    can_schedule = True
                    break
            if not can_schedule: # 如果不能调度，则跳出循环
                break
            assert new_blocks is not None

            # Schedule the request.
            scheduled_running_reqs.append(request) # 将请求加入正在运行的请求队列
            req_to_new_block_ids[request.request_id] = [ # 将新块加入请求到新块的映射
                b.block_id for b in new_blocks
            ]
            num_scheduled_tokens[request.request_id] = num_new_tokens # 将调度的令牌数量加入调度的令牌数量映射
            token_budget -= num_new_tokens # 更新调度预算
            req_index += 1

            # Encoder-related. # 编码器相关
            if encoder_inputs_to_schedule : #
                scheduled_encoder_inputs[request.request_id] = ( # 将编码器输入加入请求到新块的映射
                    encoder_inputs_to_schedule)
                # Allocate the encoder cache.  # 分配编码器缓存
                for i in encoder_inputs_to_schedule: # 遍历编码器输入
                    self.encoder_cache_manager.allocate(request, i)
                encoder_budget = new_encoder_budget

        # Record the LoRAs in scheduled_running_reqs # 记录调度后需要计算请求的LoRA数据。
        requested_loras: Set[int] = set()
        if self.lora_config:
            requested_loras = set(
                req.lora_request.lora_int_id for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0)
            assert len(requested_loras) <= self.lora_config.max_loras

        # Next, schedule the WAITING requests. 接下来调度等待的请求。
        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs: # 最大seq数是否到达最大值。
                    break

                request = self.waiting[0]

                # Check that adding the request still respects the max_loras
                # constraint. # 检查是否超过最大LoRA数量。
                if self.lora_config and request.lora_request:
                    req_lora_id = request.lora_request.lora_int_id
                    if len(requested_loras) == self.lora_config.max_loras and (
                            req_lora_id not in requested_loras):
                        # Cannot schedule.
                        # TODO (varun): This means all the other requests in
                        # the WAITING queue will be blocked by this request,
                        # even if,
                        # 1. these other requests do not use LoRA, or,
                        # 2. these other requests use the already requested
                        # LoRAs.
                        # This is too conservative and could be optimized.
                        break

                # Get already-cached tokens. # 获取已缓存的令牌。
                computed_blocks, num_computed_tokens = \
                    self.kv_cache_manager.get_computed_blocks(request)
                # Number of tokens to be scheduled.
                # We use `request.num_tokens` instead of
                # `request.num_prompt_tokens` to consider the resumed requests,
                # which have output tokens.
                num_new_tokens = request.num_tokens - num_computed_tokens # 减去已经计算过的token。
                if num_new_tokens == 0:
                    # This happens when prompt length is divisible by the block
                    # size and all blocks are cached. Now we force to recompute
                    # the last block. Note that we have to re-compute an entire
                    # block because allocate_slots() assumes num_computed_tokens
                    # is always a multiple of the block size. This limitation
                    # can potentially be removed in the future to slightly
                    # improve the performance.
                    num_computed_tokens -= self.block_size
                    num_new_tokens = self.block_size
                    computed_blocks.pop()
                num_new_tokens = min(num_new_tokens, token_budget)
                assert num_new_tokens > 0

                # Schedule encoder inputs. # 调度encoder输入。
                (encoder_inputs_to_schedule, num_new_tokens,
                 new_encoder_budget) = self._try_schedule_encoder_inputs(
                     request, num_computed_tokens, num_new_tokens,
                     encoder_budget)
                if num_new_tokens == 0:
                    # The request cannot be scheduled.
                    break

                new_blocks = self.kv_cache_manager.allocate_slots( # 分配新块。
                    request, num_new_tokens, computed_blocks)
                if new_blocks is None:
                    # The request cannot be scheduled.
                    break

                self.waiting.popleft()
                self.running.append(request)
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(
                        f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    requested_loras.add(request.lora_request.lora_int_id)
                req_to_new_block_ids[request.request_id] = [
                    b.block_id for b in computed_blocks + new_blocks
                ]
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens

                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule)
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_budget = new_encoder_budget

        # Check if the scheduling constraints are satisfied. 检查调度约束是否满足。
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert (len(scheduled_new_reqs) + len(scheduled_resumed_reqs) +
                len(scheduled_running_reqs) <= len(self.running))

        # Get the longest common prefix among all requests in the running queue. # 获取运行队列中所有请求的最长公共前缀。
        # This can be potentially used for cascade attention. # 使用到级联解码。
        num_common_prefix_blocks = 0
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request, len(self.running)))

        # Construct the scheduler output. # 构建调度输出。
        new_reqs_data = [
            NewRequestData.from_request(req,
                                        req_to_new_block_ids[req.request_id],
                                        req.num_computed_tokens)
            for req in scheduled_new_reqs
        ]
        resumed_reqs_data = [
            self._make_cached_request_data(
                req,
                req_to_new_block_ids[req.request_id],
                req.num_computed_tokens,
                resumed_from_preemption=True,
            ) for req in scheduled_resumed_reqs
        ]
        running_reqs_data = [
            self._make_cached_request_data(
                req,
                req_to_new_block_ids[req.request_id],
                req.num_computed_tokens,
                resumed_from_preemption=False,
            ) for req in scheduled_running_reqs
        ]
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=resumed_reqs_data + running_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
        )

        self.finished_req_ids = set()
        return scheduler_output

    def _make_cached_request_data(
        self,
        request: Request,
        new_block_ids: List[int],
        num_computed_tokens: int,
        resumed_from_preemption: bool,
    ) -> "CachedRequestData":
        # OPTIMIZATION: Cache the CachedRequestData objects to avoid creating
        # them at each scheduling step.
        if request.request_id in self._cached_reqs_data:
            req_data = self._cached_reqs_data[request.request_id]
            req_data.resumed_from_preemption = resumed_from_preemption
            req_data.new_block_ids = new_block_ids
            req_data.num_computed_tokens = num_computed_tokens
        else:
            req_data = CachedRequestData.from_request(request,
                                                      resumed_from_preemption,
                                                      new_block_ids,
                                                      num_computed_tokens)
            self._cached_reqs_data[request.request_id] = req_data
        return req_data

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_budget: int,
    ) -> Tuple[List[int], int, int]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.
        确定在当前步骤中需要调度的编码器输入，并相应地更新   num_new_tokens   和编码器令牌预算。
        编码器输入将被调度，如果满足以下条件：
        • 其输出令牌与当前步骤中正在计算的令牌范围重叠，即 [num_computed_tokens, num_computed_tokens + num_new_tokens)。
        • 它尚未被计算并存储在编码器缓存中。
        • 有足够的编码器令牌预算来处理它。
        • 编码器缓存有足够的空间来存储它。如果由于缓存或预算限制而无法调度编码器输入，该方法将调整   num_new_tokens  ，以便仅调度解码器令牌，直到无法调度的编码器输入之前。
        """
        if not request.has_encoder_inputs(): # 如果没有encoder输入，则返回空列表。
            return [], num_new_tokens, encoder_budget

        encoder_inputs_to_schedule: List[int] = [] # 最终被调度的encoder输入。
        mm_positions = request.mm_positions # 便利encoder的占位符。
        assert mm_positions is not None
        assert len(mm_positions) > 0
        for i, pos_info in enumerate(mm_positions):
            start_pos = pos_info["offset"] # 位置。
            num_encoder_tokens = pos_info["length"] # 编码长度。

            # The encoder output is needed if the two ranges overlap: 
            # [num_computed_tokens, num_computed_tokens + num_new_tokens) and 
            # [start_pos, start_pos + num_encoder_tokens) 
            if start_pos >= num_computed_tokens + num_new_tokens:
                # The encoder input is not needed in this step. # 现在计算的长度还未到达起始位置。
                break
            if start_pos + num_encoder_tokens <= num_computed_tokens: # 如果计算的长度已经超过起始位置，则跳过。
                # The encoder input is already computed and stored 
                # in the decoder's KV cache.  # encoder输入已经计算并存储在decoder的kv缓存中。
                continue

            if self.encoder_cache_manager.has_cache(request, i): # 判断是否已经计算。
                # The encoder input is already computed and cached.
                continue
            if (not self.encoder_cache_manager.can_allocate(request, i)
                    or num_encoder_tokens > encoder_budget):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - num_computed_tokens
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            encoder_budget -= num_encoder_tokens # 跟新预算。
            encoder_inputs_to_schedule.append(i) # 添加到调度列表。
        return encoder_inputs_to_schedule, num_new_tokens, encoder_budget

    def update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: "ModelRunnerOutput",
    ) -> EngineCoreOutputs:
        # NOTE(woosuk): This method doesn't consider speculative decoding.
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        new_running: List[Request] = []
        outputs: List[EngineCoreOutput] = []

        # NOTE(woosuk): As len(self.running) can be up to 1K or more, the below
        # loop can be a performance bottleneck. We should do our best to avoid
        # expensive operations inside the loop.
        for request in self.running:
            req_id = request.request_id
            num_tokens_scheduled = num_scheduled_tokens.get(req_id, 0)
            if num_tokens_scheduled == 0:
                # The request was not scheduled in this step.
                new_running.append(request)
                continue

            request.num_computed_tokens += num_tokens_scheduled
            # When the request's num_computed_tokens catches up its num_tokens,
            # the request generates output tokens. Otherwise, we ignore the
            # sampler output for the request.
            assert request.num_computed_tokens <= request.num_tokens

            cached_encoder_input_ids = (
                self.encoder_cache_manager.get_cached_input_ids(request))
            # OPTIMIZATION: Avoid list(set) if the set is empty.
            if cached_encoder_input_ids:
                for input_id in list(cached_encoder_input_ids):
                    start_pos = request.mm_positions[input_id]["offset"]
                    num_tokens = request.mm_positions[input_id]["length"]
                    if start_pos + num_tokens <= request.num_computed_tokens:
                        # The encoder output is already processed and stored
                        # in the decoder's KV cache.
                        self.encoder_cache_manager.free_encoder_input(
                            request, input_id)

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)

            stopped = False
            new_logprobs = None
            new_token_ids = None

            if request.num_computed_tokens == request.num_tokens:
                req_index = model_runner_output.req_id_to_index[req_id]
                # NOTE(woosuk): Currently, we assume that each request
                # generates at most one token at each step.
                token_id = sampled_token_ids[req_index]
                request.append_output_token_ids(token_id)
                num_new_tokens = 1
                # TODO: Update the KV cache manager for prefix caching.

                # Check for stop and update request state.
                # This must be called before we make the EngineCoreOutput.
                stopped = self._check_stop(request)
                if stopped:
                    self._free_request(request)

                # Extract sample logprobs if needed.
                if request.sampling_params.logprobs is not None:
                    assert logprobs is not None
                    # NOTE: once we support N tokens per step (spec decode),
                    # the outer lists can be of length > 1.
                    new_logprobs = logprobs.slice(req_index, req_index + 1)

                new_token_ids = request.output_token_ids[-num_new_tokens:]

            # Transmit partial if chunked prefill & prompt logprobs is enabled
            if new_token_ids or prompt_logprobs_tensors is not None:
                # Add EngineCoreOutput for this Request.
                outputs.append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids or [],
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        stop_reason=request.stop_reason))

            if not stopped:
                new_running.append(request)

        self.running = new_running
        return EngineCoreOutputs(
            outputs=outputs,
            scheduler_stats=self.make_stats(),
        )

    # 检查是否停止。
    def _check_stop(self, request: Request) -> bool:
        if (request.num_tokens >= self.max_model_len 
                or request.num_output_tokens >= request.max_tokens): # 检查是否达到最大长度。
            request.status = RequestStatus.FINISHED_LENGTH_CAPPED
            return True

        sampling_params = request.sampling_params
        last_token_id = request.output_token_ids[-1] # 获取最后一个token是否命中停止id或者eos。 
        if (not sampling_params.ignore_eos
                and last_token_id == request.eos_token_id):
            request.status = RequestStatus.FINISHED_STOPPED
            return True

        if last_token_id in (sampling_params.stop_token_ids or ()):
            request.status = RequestStatus.FINISHED_STOPPED
            request.stop_reason = last_token_id
            return True
        return False

    def add_request(self, request: Request) -> None:
        self.waiting.append(request) # 等待队列。
        self.requests[request.request_id] = request 

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: RequestStatus,
    ) -> None:
        """Handles the finish signal from outside the scheduler.
        
        For example, the API server can abort a request when the client
        disconnects.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids, )
        request_ids = set(request_ids) # 

        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None:
                # Invalid request ID.
                continue

            if request.status == RequestStatus.RUNNING:
                self.running.remove(request)
            else:
                self.waiting.remove(request)
            request.status = finished_status
            self._free_request(request)

    def _free_request(self, request: Request) -> None:
        assert request.is_finished()
        self.kv_cache_manager.free(request)
        self.kv_cache_manager.free_block_hashes(request)
        self.encoder_cache_manager.free(request)
        self._cached_reqs_data.pop(request.request_id, None)
        del self.requests[request.request_id]
        self.finished_req_ids.add(request.request_id)

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def has_unfinished_requests(self) -> bool:
        return self.get_num_unfinished_requests() > 0

    def reset_prefix_cache(self) -> bool:
        return self.kv_cache_manager.reset_prefix_cache()

    def make_stats(self) -> SchedulerStats:
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            gpu_cache_usage=self.kv_cache_manager.usage,
        )


@dataclass
class NewRequestData:

    req_id: str
    prompt_token_ids: List[int]
    prompt: Optional[str]
    mm_inputs: List["MultiModalKwargs"]
    mm_hashes: List[str]
    mm_positions: List["PlaceholderRange"]
    sampling_params: SamplingParams
    block_ids: List[int]
    num_computed_tokens: int
    lora_request: Optional[LoRARequest]

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: List[int],
        num_computed_tokens: int,
    ) -> "NewRequestData":
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            prompt=request.prompt,
            mm_inputs=request.mm_inputs,
            mm_hashes=request.mm_hashes,
            mm_positions=request.mm_positions,
            sampling_params=request.sampling_params,
            block_ids=block_ids,
            num_computed_tokens=num_computed_tokens,
            lora_request=request.lora_request,
        )


@dataclass
class CachedRequestData:

    req_id: str
    # If resumed_from_preemption is False, new_block_ids will be appended to
    # the request's block IDs. If True, new_block_ids will be used as the
    # request's block IDs instead of appending to the existing block IDs.
    resumed_from_preemption: bool
    new_block_ids: List[int]
    num_computed_tokens: int

    @classmethod # 数据类的初始化写法。
    def from_request(
        cls,
        request: Request,
        resumed_from_preemption: bool,
        new_block_ids: List[int],
        num_computed_tokens: int,
    ) -> "CachedRequestData":
        return cls(
            req_id=request.request_id,
            resumed_from_preemption=resumed_from_preemption,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
        )


@dataclass
class SchedulerOutput:

    scheduled_new_reqs: List[NewRequestData] # 调度后的请求。 
    scheduled_cached_reqs: List[CachedRequestData] # 调度后的缓存请求。

    num_scheduled_tokens: Dict[str, int] # 调度的token数量。
    total_num_scheduled_tokens: int # 调度的总token数量。
    scheduled_encoder_inputs: Dict[str, List[int]] # 调度的encoder输入。
    num_common_prefix_blocks: int # 公共前缀块的数量。

    finished_req_ids: Set[str] # 完成的请求id。
    free_encoder_input_ids: List[Tuple[str, int]] # 释放的encoder输入id。
