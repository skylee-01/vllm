# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Optional

from vllm.distributed.kv_events import KVCacheEvent
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


@dataclass
class KVCacheBlocks:
    """
    KVCacheBlocks 是 KVCacheManager 的分配结果，作为调度器和 KVCacheManager 之间的接口，
    以隐藏 KVCacheManager 的内部数据结构。

    这个类封装了一组 KV 缓存块，通常代表一个或多个请求在当前调度步骤中将使用的 KV 缓存空间。
    """
    blocks: tuple[list[KVCacheBlock], ...]
    """
    blocks[i][j] 指的是第 i 个 kv_cache_group 中的第 j 个 token 块。
    我们不使用 token 块作为外层维度，因为它假设所有 kv_cache_groups 具有相同数量的块，
    这目前是正确的，但如果未来我们希望为不同的 kv_cache_groups 提供不同的 block_size，
    这种假设将会被打破。
    """

    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        """
        将两个 KVCacheBlocks 实例相加。

        此方法用于合并两个 KVCacheBlocks 实例，例如在 KV 传输场景中合并本地和远程的 KV 块。
        它通过按 KV 缓存组对应地连接每个组中的块列表来完成合并。

        Args:
            other (KVCacheBlocks): 要与当前实例合并的另一个 KVCacheBlocks 实例。

        Returns:
            KVCacheBlocks: 合并后的新 KVCacheBlocks 实例。
        """
        return KVCacheBlocks(
            tuple(blk1 + blk2
                  for blk1, blk2 in zip(self.blocks, other.blocks)))

    def get_block_ids(self) -> tuple[list[int], ...]:
        """
        将 KVCacheBlocks 实例转换为块 ID。

        此方法从每个 KVCacheBlock 对象中提取其 `block_id`，并将它们组织成与 `blocks`
        属性相同的结构（即，一个元组，其中包含每个 KV 缓存组的块 ID 列表）。

        Returns:
            tuple[list[int], ...]: 一个元组列表，其中：
            * 外层元组对应于 KV 缓存组。
            * 每个内层列表包含该组中块的 `block_id`。
        """
        return tuple([blk.block_id for blk in group] for group in self.blocks)

    def get_unhashed_block_ids(self) -> list[int]:
        """
        从 KVCacheBlocks 实例中获取未哈希块的 block_ids。

        此方法用于在分层缓存或分布式 KV 缓存场景中，识别那些尚未被哈希处理或持久化的块。
        它遍历所有块，并返回那些 `block_hash` 为 None 的块的 `block_id`。

        Returns:
            list[int]: 未哈希块的 `block_id` 列表。

        Raises:
            AssertionError: 如果有多个 KV 缓存组，则会触发断言错误，因为目前只支持单个组。
        """
        assert len(self.blocks) == 1, "Only one group is supported"
        return [
            block.block_id for block in self.blocks[0]
            if block.block_hash is None
        ]

    def new_empty(self) -> "KVCacheBlocks":
        """
        创建一个不包含任何块的新的 KVCacheBlocks 实例。

        此方法通常用于初始化一个空的 KVCacheBlocks 对象，例如在开始调度新请求或重置缓存状态时。
        它会根据现有的 KV 缓存组数量创建一个空的块列表元组。

        Returns:
            KVCacheBlocks: 一个不包含任何块的新 KVCacheBlocks 实例。
        """
        return KVCacheBlocks(tuple([] for _ in range(len(self.blocks))))


class KVCacheManager:
    """
    KVCacheManager 负责管理 KV 缓存块的分配、释放和重用。
    它作为调度器和底层 KV 缓存存储之间的接口，处理所有与 KV 缓存相关的操作。
    该管理器支持前缀缓存、推测解码和多 KV 缓存组。
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
    ) -> None:
        """
        初始化 KVCacheManager 实例。

        Args:
            kv_cache_config (KVCacheConfig): KV 缓存的配置，包括块大小和 KV 缓存组信息。
            max_model_len (int): 模型的最大序列长度。
            enable_caching (bool): 是否启用前缀缓存。默认为 True。
            use_eagle (bool): 是否使用 Eagle 推测解码模式。默认为 False。
            log_stats (bool): 是否记录统计信息。默认为 False。
            enable_kv_cache_events (bool): 是否启用 KV 缓存事件。默认为 False。
        """
        self.max_model_len = max_model_len

        self.enable_caching = enable_caching
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        # FIXME: 使前缀缓存统计信息取决于 log_stats
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        self.block_size: Optional[int] = None
        if self.enable_caching:
            assert len(
                set(g.kv_cache_spec.block_size
                    for g in kv_cache_config.kv_cache_groups)
            ) == 1, "Only one block size is supported for now"
            self.block_size = kv_cache_config.kv_cache_groups[
                0].kv_cache_spec.block_size

        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            use_eagle=self.use_eagle,
            enable_caching=self.enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
        )
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_pool = self.coordinator.block_pool
        self.kv_cache_config = kv_cache_config

    @property
    def usage(self) -> float:
        """
        获取 KV 缓存的使用情况。

        Returns:
            float: KV 缓存使用率（0.0 到 1.0 之间）。
        """
        return self.block_pool.get_usage()

    def make_prefix_cache_stats(self) -> Optional[PrefixCacheStats]:
        """
        获取（并重置）前缀缓存统计信息。

        Returns:
            Optional[PrefixCacheStats]: 当前前缀缓存统计信息，如果禁用日志记录则为 None。
        """
        if not self.log_stats:
            return None
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats() # 重置统计计数器
        return stats

    def get_computed_blocks(self,
                            request: Request) -> tuple[KVCacheBlocks, int]:
        """
        获取请求的已计算（缓存）块。
        请注意，已计算的块必须是完整的。

        Args:
            request (Request): 要获取已计算块的请求。

        Returns:
            tuple[KVCacheBlocks, int]: 一个包含以下内容的元组：
                - KVCacheBlocks: 请求已计算的块列表。
                - int: 已计算的 token 数量。
        """
        # 前缀缓存被禁用，或者当请求需要 prompt logprobs 时，我们跳过前缀缓存。
        if (not self.enable_caching
                or (request.sampling_params is not None
                    and request.sampling_params.prompt_logprobs is not None)):
            return self.create_empty_block_list(), 0

        # NOTE: 当所有 token 都命中缓存时，我们必须重新计算最后一个 token
        # 以获取 logits。因此，将 max_cache_hit_length 设置为 prompt_length - 1。
        # 这可能触发整个块的重新计算，而不仅仅是单个最后一个 token，因为 allocate_slots()
        # 要求 num_computed_tokens 与块大小对齐。未来移除此限制可以略微提高性能。
        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens = (
            self.coordinator.find_longest_cache_hit(request.block_hashes,
                                                    max_cache_hit_length))

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.requests += 1
            self.prefix_cache_stats.queries += request.num_tokens
            self.prefix_cache_stats.hits += num_new_computed_tokens

        return KVCacheBlocks(computed_blocks), num_new_computed_tokens

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: Optional[KVCacheBlocks] = None,
        num_lookahead_tokens: int = 0,
        delay_cache_blocks: bool = False,
    ) -> Optional[KVCacheBlocks]:
        """
        为需要追加新 token 的请求添加槽位。

        Args:
            request (Request): 要分配槽位的请求。
            num_new_tokens (int): 要分配的 token 数量，包括外部 token。
                请注意，这不包括已在本地计算的 token（即 new_computed_blocks）。
            num_new_computed_tokens (int): 刚命中前缀缓存的新计算 token 数量，不包括外部 token。
            new_computed_blocks (Optional[KVCacheBlocks]): 上述新计算 token 的缓存块。
            num_lookahead_tokens (int): 要分配的推测 token 数量。
                这由带有 KV 缓存的推测解码提议器（例如 eagle）使用。
            delay_cache_blocks (bool): 是否跳过缓存块。这在 P/D 分配用于
                未来步骤中将完成的 KV 传输的块时使用。

        块布局 (Block Layout):
        ```
        -----------------------------------------------------------------------
        | < computed > | < new computed > |    < new >    | < pre-allocated > |
        -----------------------------------------------------------------------
        |                  < required >                   |
        ---------------------------------------------------
        |                    < full >                  |
        ------------------------------------------------
                                          | <new full> |
                                          --------------
        ```
        下面的 `*_blocks` 在此布局中进行了说明。

        Returns:
            Optional[KVCacheBlocks]: 新分配的块列表，如果无法分配则为 None。
        
        Raises:
            ValueError: 如果 `num_new_tokens` 为 0 则抛出。
        """
        if num_new_tokens == 0:
            raise ValueError("num_new_tokens must be greater than 0")

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = tuple(
                [] for _ in range(len(self.kv_cache_config.kv_cache_groups))) # 为每个 KV 缓存组创建一个空列表

        # 释放注意力计算过程中跳过的块 (例如，滑动窗口外的 token)。
        # 即使由于可用块不足而无法调度此请求，也可以执行此操作。
        # 应在分配新块之前调用此函数，以减少被逐出块的数量。
        self.coordinator.remove_skipped_blocks(request.request_id,
                                               request.num_computed_tokens)

        # 已计算 token 的数量是已计算 token 加上新前缀缓存命中数
        num_computed_tokens = (request.num_computed_tokens +
                               num_new_computed_tokens)
        # 需要槽位的 token 总数，限制在模型最大长度之内
        num_tokens_need_slot = min(
            num_computed_tokens + num_new_tokens + num_lookahead_tokens,
            self.max_model_len)

        # 获取需要分配的块数量
        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
        )

        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            # 无法分配新块，因为空闲块不足
            return None

        # 触碰已计算的块以确保它们不会被逐出。
        if self.enable_caching:
            self.block_pool.touch(new_computed_block_list)
        else:
            assert not any(new_computed_block_list), (
                "Computed blocks should be empty when "
                "prefix caching is disabled")

        # 将新计算的块追加到请求块中，以避免新块无法分配的情况。
        self.coordinator.save_new_computed_blocks(request.request_id,
                                                  new_computed_block_list)

        # 分配新的块
        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id, num_tokens_need_slot)

        # P/D: 如果需要从远程接收，则延迟缓存块。更新本地缓存块的状态。
        if not self.enable_caching or delay_cache_blocks:
            return KVCacheBlocks(new_blocks)

        # NOTE(woosuk): 我们希望提交 (缓存) 最多 num_computed_tokens + num_new_tokens，
        # 但必须排除“不可提交”的 token (例如，可能被拒绝的草稿 token)。
        # 因此，我们将数量限制在 `request.num_tokens`，确保只缓存“最终确定”的 token。
        num_tokens_to_cache = min(num_computed_tokens + num_new_tokens,
                                  request.num_tokens)
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        return KVCacheBlocks(new_blocks)

    def free(self, request: Request) -> None:
        """
        释放为请求分配的块。
        我们以相反的顺序释放块，以便在启用缓存时首先逐出尾部块。

        Args:
            request (Request): 要释放块的请求。
        """
        self.coordinator.free(request.request_id)

    def reset_prefix_cache(self) -> bool:
        """
        重置前缀缓存。此函数可用于 RLHF 流中，以在权重更新后使前缀缓存失效，
        或用于重置前缀缓存状态以进行基准测试。

        Returns:
            bool: 如果前缀缓存成功重置，则为 True；否则为 False。
        """
        if not self.block_pool.reset_prefix_cache():
            return False
        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.reset = True
        return True

    def get_num_common_prefix_blocks(
        self,
        request: Request,
        num_running_requests: int,
    ) -> list[int]:
        """
        计算每个 KV 缓存组中所有处于 RUNNING 状态的请求共享的公共前缀块的数量。

        此函数通过选择任意请求并遍历其块来确定此数量。如果块的 `ref_cnt` 等于
        处于 RUNNING 状态的请求总数，则该块被视为公共前缀块。

        NOTE(woosuk): 处于 RUNNING 状态的请求数量 **大于或等于** 当前步中调度的请求数量。
        这是因为 RUNNING 状态仅表示：
        1. 请求尚未完成，并且
        2. 请求持有其未释放的块。

        虽然所有已调度的请求都必须处于 RUNNING 状态，但反之则不一定成立。
        可能存在当前步中未调度的 RUNNING 请求。

        这可能导致一种边缘情况，即即使所有已调度的请求共享一个公共前缀，
        公共前缀块的数量也为 0。这发生在可能存在不共享公共前缀的未调度 RUNNING 请求时。
        目前，这种情况无法轻易检测到，因此在这种情况下函数返回 0。

        Args:
            request (Request): 处于 RUNNING 状态的任意请求，用于识别公共前缀块。
            num_running_requests (int): 处于 RUNNING 状态的请求总数。
                这可能与当前步中调度的请求数量不同。

        Returns:
            list[int]: 每个 KV 缓存组的公共前缀块的数量。
        """
        assert request.status == RequestStatus.RUNNING
        return self.coordinator.get_num_common_prefix_blocks(
            request.request_id, num_running_requests)

    def take_events(self) -> list[KVCacheEvent]:
        """
        从块池中获取 KV 缓存事件。

        此方法用于收集所有在当前步中发生的 KV 缓存相关事件，例如块的分配、释放或重用。

        Returns:
            list[KVCacheEvent]: KV 缓存事件列表。
        """
        return self.block_pool.take_events()

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        """
        获取请求的块 ID。

        此方法通过协调器获取请求的块，然后使用 KVCacheBlocks 实例将其转换为块 ID 元组。

        Args:
            request_id (str): 请求的唯一标识符。

        Returns:
            tuple[list[int], ...]: 一个元组列表，其中包含请求的每个 KV 缓存组的块 ID。
        """
        return KVCacheBlocks(
            self.coordinator.get_blocks(request_id)).get_block_ids()

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """
        如果启用，则缓存请求的块。

        此方法将请求的已计算 token 对应的块标记为已缓存，以实现前缀缓存。
        只有在 `enable_caching` 为 True 时才执行缓存操作。

        Args:
            request (Request): 要缓存块的请求。
            num_computed_tokens (int): 已计算的 token 数量，用于确定缓存范围。
        """
        if self.enable_caching:
            self.coordinator.cache_blocks(request, num_computed_tokens)

    def create_empty_block_list(self) -> KVCacheBlocks:
        """
        创建一个不包含任何块的新的 KVCacheBlocks 实例。

        此方法用于在不需要任何 KV 缓存块时（例如，前缀缓存被禁用时）提供一个空的 KVCacheBlocks 对象。

        Returns:
            KVCacheBlocks: 一个不包含任何块的新 KVCacheBlocks 实例。
        """
        return KVCacheBlocks(tuple([]
                                   for _ in range(self.num_kv_cache_groups)))
