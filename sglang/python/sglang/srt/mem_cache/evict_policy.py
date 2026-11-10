from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Union

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode


class EvictionStrategy(ABC):
    @abstractmethod
    def get_priority(self, node: "TreeNode") -> Union[float, Tuple]:
        pass


class TypedOffloadingStrategy(EvictionStrategy):
    def __init__(
        self,
        base_strategy: EvictionStrategy,
        get_program_type_fn: Callable[[Optional[str]], int],
        eviction_order_map: Dict[int, int],
    ):
        self._base = base_strategy
        self._get_program_type = get_program_type_fn
        self._order_map = eviction_order_map

    def get_priority(self, node: "TreeNode") -> Tuple:
        base_priority = self._base.get_priority(node)
        pt = self._get_program_type(node.program_id)
        type_key = self._order_map.get(pt, int(pt))
        if isinstance(base_priority, tuple):
            return (type_key,) + base_priority
        return (type_key, base_priority)


class TypedHostEvictionStrategy(EvictionStrategy):
    def __init__(
        self,
        base_strategy: EvictionStrategy,
        get_program_type_fn: Callable[[Optional[str]], int],
        eviction_order_map: Dict[int, int],
        interactive_type: int,
        async_type: int,
        epoch_fn: Callable[[], int],
        active_fn: Callable[[], bool],
    ):
        self._base = base_strategy
        self._get_program_type = get_program_type_fn
        self._order_map = eviction_order_map
        self._interactive = interactive_type
        self._lifted_key = eviction_order_map[async_type] + 0.5
        self._epoch_fn = epoch_fn
        self._active_fn = active_fn

    def get_priority(self, node: "TreeNode") -> Tuple:
        base_priority = self._base.get_priority(node)
        if not isinstance(base_priority, tuple):
            base_priority = (base_priority,)
        pt = self._get_program_type(node.program_id)
        type_key = self._order_map.get(pt, int(pt))
        if not self._active_fn():
            return (0, type_key, 0) + base_priority
        if pt == self._interactive and node.value is None:
            type_key = max(type_key, self._lifted_key)
        epoch = self._epoch_fn()
        need = 1 if getattr(node, "load_need_epoch", -1) == epoch else 0
        written = 1 if getattr(node, "write_back_epoch", -1) == epoch else 0
        return (need, type_key, written) + base_priority


class LRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return node.last_access_time


class LFUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        return (node.hit_count, node.last_access_time)


class FIFOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return node.creation_time


class MRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return -node.last_access_time


class FILOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return -node.creation_time


class PriorityStrategy(EvictionStrategy):
    """Priority-aware eviction: lower priority values evicted first, then LRU within same priority."""

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        # Return (priority, last_access_time) so lower priority nodes are evicted first
        return (node.priority, node.last_access_time)


class SLRUStrategy(EvictionStrategy):
    def __init__(self, protected_threshold: int = 2):
        self.protected_threshold = protected_threshold

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        # Priority Logic:
        # Smaller value = Evicted earlier.
        #
        # Segment 0 (Probationary): hit_count < threshold
        # Segment 1 (Protected): hit_count >= threshold
        #
        # Tuple comparison: (segment, last_access_time)
        # Nodes in segment 0 will always be evicted before segment 1.
        # Inside the same segment, older nodes (smaller time) are evicted first.

        is_protected = 1 if node.hit_count >= self.protected_threshold else 0
        return (is_protected, node.last_access_time)
