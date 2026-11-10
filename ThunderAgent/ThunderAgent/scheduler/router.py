"""Router with program state tracking - supports multiple backends."""
import asyncio
import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Awaitable, Tuple

import httpx
from fastapi.responses import Response

from ..backend import BackendState, MetricsClient, SGLangMetricsClient, VLLMMetricsClient
from ..backend.skyrl_metrics import SkyRLMetricsClient
from ..program import Program, ProgramStatus, ProgramState
from ..profile.state import ProfileState
from ..config import get_config
from .vllm_request_processor import (
    forward_streaming_request,
    forward_non_streaming_request,
    forward_get_request,
)

logger = logging.getLogger(__name__)


@dataclass
class PausedInfo:
    """Metadata for a paused program stored in the global paused pool."""
    program_id: str
    total_tokens: int
    paused_at: float
    origin_backend: Optional[str]  # None for new programs that haven't been assigned yet
    step_count: int


PROGRAM_TYPE_INTERACTIVE = "INTERACTIVE"
PROGRAM_TYPE_ASYNC = "ASYNC"
PROGRAM_TYPE_INACTIVE = "INACTIVE"

IDLE_DISPLACEMENT_MAX_FRAC = 0.5

PREFIX_GROUP_SYSTEM_CHARS = 1024


def _extract_system_text(payload: Dict[str, Any]) -> str:
    sys_parts: List[str] = []
    for m in (payload.get("messages") or []):
        if isinstance(m, dict) and m.get("role") == "system":
            c = m.get("content")
            if isinstance(c, str):
                sys_parts.append(c)
            elif isinstance(c, list):
                for b in c:
                    if isinstance(b, dict) and isinstance(b.get("text"), str):
                        sys_parts.append(b["text"])
    return "\x00".join(sys_parts)


def _prefix_group_key(payload: Dict[str, Any]) -> str:
    tools = payload.get("tools")
    if tools is None:
        eb = payload.get("extra_body")
        if isinstance(eb, dict):
            tools = eb.get("tools")
    try:
        tools_str = (json.dumps(tools, sort_keys=True, ensure_ascii=False)
                     if tools else "")
    except (TypeError, ValueError):
        tools_str = str(tools)
    sys_str = _extract_system_text(payload)[:PREFIX_GROUP_SYSTEM_CHARS]
    if not sys_str and not tools_str:
        return "_empty_"
    blob = sys_str + "\x01" + tools_str
    return hashlib.sha1(blob.encode("utf-8", "ignore")).hexdigest()[:16]


class MultiBackendRouter:
    """Router with program state tracking, supports multiple backends."""

    @staticmethod
    def _create_metrics_client(backend_type: str, url: str) -> MetricsClient:
        """Create a metrics client for the given backend type.
        
        Args:
            backend_type: Backend type ("vllm" or "sglang")
            url: Backend base URL
        
        Returns:
            MetricsClient implementation for the backend type
        """
        if backend_type == "vllm":
            return VLLMMetricsClient(url)
        if backend_type == "sglang":
            return SGLangMetricsClient(url)
        if backend_type == "skyrl":
            return SkyRLMetricsClient(url)
        raise ValueError(f"Unsupported backend_type: {backend_type}")

    def __init__(
        self, 
        backend_urls: str | List[str], 
        *, 
        profile_enabled: bool = False,
        scheduling_enabled: bool = True,
        scheduler_interval: float = 5.0,
        backend_type: str = "vllm",
        acting_token_weight: float = 1.0,
        use_acting_token_decay: bool = False,
    ) -> None:
        # Support single URL string or list of URLs
        if isinstance(backend_urls, str):
            backend_urls = [url.strip() for url in backend_urls.split(",") if url.strip()]
        if scheduling_enabled and backend_type != "sglang":
            raise ValueError(
                "MORI scheduling requires --backend-type sglang "
                "(SGLang with HiCache and typed offloading)"
            )
        
        # Weight for acting tokens in capacity calculation
        self.acting_token_weight = acting_token_weight
        
        # All backends (pass acting_token_weight as tool_coefficient)
        self.backends: Dict[str, BackendState] = {}
        for url in backend_urls:
            metrics_client = self._create_metrics_client(backend_type, url)
            self.backends[url] = BackendState(
                url=url,
                tool_coefficient=acting_token_weight,
                metrics_client=metrics_client,
                use_acting_token_decay=use_acting_token_decay,
            )
        
        # All programs (single source of truth)
        # Key: program_id, Value: Program (which includes backend_url)
        self.programs: Dict[str, Program] = {}

        # Global paused pool shared across backends (program_id -> PausedInfo).
        self.global_waiting_queue: Dict[str, PausedInfo] = {}

        # Lock for atomic claim (select + pop) from global_waiting_queue.
        self.pause_resume_lock = asyncio.Lock()
        
        # Profile configuration
        self.profile_enabled = profile_enabled
        
        # Scheduling mode: True = "tr" (capacity scheduling), False = "default" (pure proxy)
        self.scheduling_enabled = scheduling_enabled

        self.client = httpx.AsyncClient(
            timeout=900.0,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        )
        
        # Scheduler task for periodic capacity check
        self._scheduler_task: Optional[asyncio.Task] = None
        self._scheduler_stop = False
        self._scheduler_interval = scheduler_interval
        
        # Global char-to-token ratio for token estimation
        # Initial value is 5.0 (1 token ≈ 5 chars), updated with momentum after each request
        self.char_to_token_ratio: float = 5.0
        self._ratio_initialized: bool = False

        self._warm_queues: Dict[str, Dict[str, int]] = {url: {} for url in self.backends}
        self._group_prefix_len: Dict[str, float] = {}
        self._first_cached_tokens: Dict[str, int] = {}
        self._pending_mark_tasks: List[asyncio.Task] = []

    async def start(self):
        """Start the router."""
        logger.info(f"Started router with {len(self.backends)} backend(s): {list(self.backends.keys())}")
        
        # Always fetch cache config (needed for active_program_tokens_ratio)
        for backend in self.backends.values():
            await backend.fetch_cache_config()
        
        # Start metrics monitoring on each backend if enabled
        config = get_config()
        if config.metrics_enabled or self.scheduling_enabled:
            for backend in self.backends.values():
                await backend.start_monitoring(config.metrics_interval)
        
        # Start the periodic scheduler if scheduling is enabled
        if self.scheduling_enabled:
            self._scheduler_stop = False
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
            logger.info(f"Started scheduler loop (interval={self._scheduler_interval}s)")

    async def stop(self):
        """Stop the router."""
        # Stop the scheduler
        if self._scheduler_task:
            self._scheduler_stop = True
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
            self._scheduler_task = None
            logger.info("Stopped scheduler loop")
        
        # Stop metrics monitoring on each backend
        for backend in self.backends.values():
            await backend.stop_monitoring()
        
        await self.client.aclose()
        logger.info("Router stopped")

    # -------------------------------------------------------------------------
    # Backend Selection
    # -------------------------------------------------------------------------

    def get_backend(self, url: str) -> Optional[BackendState]:
        """Get a backend by URL."""
        return self.backends.get(url)

    def get_default_backend(self) -> BackendState:
        """Get the first backend (for simple single-backend usage)."""
        return next(iter(self.backends.values()))

    def select_backend_for_new_program_default(self) -> BackendState:
        """Select the least loaded backend for a new program."""
        # Count programs per backend
        #### TODO change backend assign logistics
        backend_load: Dict[str, int] = {url: 0 for url in self.backends}
        for state in self.programs.values():
            if state.backend_url in backend_load:
                backend_load[state.backend_url] += 1
        
        # Find the backend with least programs (only consider healthy ones)
        min_load = float('inf')
        best_backend = None
        for url, load in backend_load.items():
            backend = self.backends[url]
            if backend.healthy and load < min_load:
                min_load = load
                best_backend = backend
        
        # Fallback to first backend if all unhealthy
        return best_backend or self.get_default_backend()

    # -------------------------------------------------------------------------
    # Program State Management
    # -------------------------------------------------------------------------

    @staticmethod
    def _estimate_system_prompt_tokens(payload: Dict[str, Any]) -> int:
        """Estimate system prompt tokens from the first request payload."""
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return 0
        
        parts: List[str] = []
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "system":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    text = item.get("text") or item.get("input_text")
                    if isinstance(text, str):
                        parts.append(text)
        
        if not parts:
            return 0
        
        text = "\n".join(parts)
        return max(0, len(text) // 5)

    def get_or_create_program(self, program_id: str) -> Program:
        """Get existing program or create new one.
        
        Only creates the program and estimates token count.
        Backend assignment is deferred to update_program_before_request.
        
        Args:
            program_id: Unique identifier for the program
            payload: Request payload, used to estimate token count for new programs
        """
        if program_id not in self.programs:
            profile = ProfileState(program_id=program_id) if self.profile_enabled else None
            state = Program(
                program_id=program_id,
                backend_url=None,
                status=ProgramStatus.REASONING,  # Request arrived, program is reasoning
                state=ProgramState.ACTIVE,
                profile=profile,
            )
            # Token estimation is done in update_program_before_request
            self.programs[program_id] = state
            logger.debug(f"Created program {program_id}")
        
        return self.programs[program_id]
    
    def _select_backend_for_new_program(self, estimated_tokens: int = 0) -> Optional[str]:
        """Select a backend for a new program (TR mode only).
        
        This function is only called when scheduling_enabled=True.
        
        Args:
            estimated_tokens: Estimated token count for the new program
        
        Returns:
            backend_url if can assign directly (queue empty + has capacity)
            None if should wait in queue
        """
        from ..backend.state import BUFFER_PER_PROGRAM
        
        # Only assign if queue is empty and backend has capacity
        if self._parked_returner_fits():
            logger.info("[MORI] new program deferred: a parked returner fits (tokens=%d)",
                        estimated_tokens)
            return None

        # Find backend with least active tokens that has capacity for this program
        best_backend = None
        min_tokens = float('inf')
        required_capacity = estimated_tokens + BUFFER_PER_PROGRAM
        
        for backend in self.backends.values():
            if not backend.healthy:
                continue
            if backend.remaining_capacity() < required_capacity:
                continue  # Not enough capacity for this program
            if backend.active_program_tokens < min_tokens:
                min_tokens = backend.active_program_tokens
                best_backend = backend
        
        return best_backend.url if best_backend else None

    def get_backend_for_program(self, program_id: str) -> BackendState:
        """Get the backend assigned to a program."""
        state = self.programs.get(program_id)
        if state and state.backend_url in self.backends:
            return self.backends[state.backend_url]
        return self.get_default_backend()

    async def update_program_before_request(self, program_id: str, state: Program, payload: Dict[str, Any]) -> bool:
        """Update program state before sending request to vLLM.
        
        If scheduling_enabled=False (default mode): pure proxy, no capacity checks.
        If scheduling_enabled=True (tr mode): wait for scheduler to resume if PAUSED.
        
        Returns: True if can proceed
        """
        state.step_count += 1
        is_new_program = state.step_count == 1
        
        # Update context_len and estimate total_tokens using char_to_token_ratio
        state.context_len = len(json.dumps(payload, ensure_ascii=False))
        state.total_tokens = int(state.context_len / self.char_to_token_ratio)

        if state.prefix_group_key is None:
            state.prefix_group_key = _prefix_group_key(payload)
        
        # ---------------------------------------------------------------------
        # Default mode: pure proxy, no scheduling
        # ---------------------------------------------------------------------
        if not self.scheduling_enabled:
            backend = self.backends.get(state.backend_url)
            if not backend:
                # Assign to least loaded backend
                backend = self.select_backend_for_new_program_default()
                state.backend_url = backend.url
            
            if is_new_program:
                backend.register_program(program_id, state)
            # Status change is enough - token stats are computed from program status
            state.status = ProgramStatus.REASONING
            state.acting_since = None
            return True
        
        # ---------------------------------------------------------------------
        # TR mode: scheduler-based capacity management
        # ---------------------------------------------------------------------
        # Step 1: Handle PAUSED programs (wait for resume)
        # Note: waiting_event is created in _pause_program and cleared in _resume_program.
        # If waiting_event is not None, the program is still PAUSED and we must wait.
        # Set status to REASONING before waiting so the scheduler sees the correct
        # priority: REASONING programs (pending request) > ACTING programs (idle).
        if state.waiting_event is not None:
            self._set_reasoning(state)
            if not await self._try_promote_on_arrival(program_id, state):
                await self._wait_for_resume(program_id, state)
            # After _wait_for_resume returns, _resume_program has been called which:
            # - Registered program with target backend
            # - Set state.state = ACTIVE
            # - Cleared waiting_event = None
            # Fall through to Step 3
        
        # Step 2: Handle new programs (assign backend or queue)
        if is_new_program and state.backend_url is None:
            backend_url = self._select_backend_for_new_program(state.total_tokens)
            if backend_url:
                # Direct assignment: register program with backend
                state.backend_url = backend_url
                backend = self.backends[backend_url]
                backend.register_program(program_id, state)
                logger.debug(f"Assigned new program {program_id} to {backend_url}")
                self._set_reasoning(state)
                state.reasoning_run_start = time.time()
                return True
            else:
                # Queue and wait — set REASONING before waiting for priority
                self._set_reasoning(state)
                state.waiting_event = asyncio.Event()
                state.state = ProgramState.PAUSED
                self._add_to_global_waiting_queue_sync(program_id, state, backend=None)
                logger.debug(f"Queued new program {program_id} (tokens={state.total_tokens})")
                await self._wait_for_resume(program_id, state)
                # After resume: program registered with backend by _resume_program
                state.reasoning_run_start = time.time()
                return True
        
        # Step 3: Normal case - existing ACTIVE program with backend
        backend = self.backends.get(state.backend_url)
        if not backend:
            logger.error(f"Program {program_id} has no valid backend")
            return False
        
        self._set_reasoning(state)
        state.reasoning_run_start = time.time()
        return True

    def _set_reasoning(self, state: Program) -> None:
        if state.status == ProgramStatus.ACTING and state.acting_since is not None:
            state.recent_acting_durations.append(max(0.0, time.time() - state.acting_since))
        state.status = ProgramStatus.REASONING
        state.acting_since = None

    def update_program_after_request(
        self, program_id: str, state: Program, total_tokens: int, prompt_tokens: int = 0
    ) -> None:
        """Update program state after receiving response from vLLM.
        
        Transitions to ACTING (off GPU, executing tool).
        Updates token counts. If marked for pause, pause immediately.
        Also updates the global char_to_token_ratio for future token estimation.
        
        Args:
            program_id: The program ID
            state: The program state
            total_tokens: Total tokens from vLLM response (prompt + completion)
            prompt_tokens: Prompt/prefill tokens from vLLM response
        """
        if state.reasoning_run_start is not None:
            state.recent_reasoning_durations.append(max(0.0, time.time() - state.reasoning_run_start))
            state.reasoning_run_start = None

        # Transition to ACTING
        state.status = ProgramStatus.ACTING
        state.acting_since = time.time()
        
        # Update global char_to_token_ratio based on actual prefill
        # ratio = context_len / prompt_tokens (chars per token)
        if prompt_tokens > 0 and state.context_len > 0:
            current_ratio = state.context_len / prompt_tokens
            if not self._ratio_initialized:
                # First request: directly assign
                self.char_to_token_ratio = current_ratio
                self._ratio_initialized = True
                logger.debug(f"Initialized char_to_token_ratio={self.char_to_token_ratio:.2f}")
            else:
                # Subsequent requests: momentum update (0.2 new + 0.8 old)
                self.char_to_token_ratio = 0.2 * current_ratio + 0.8 * self.char_to_token_ratio
                logger.debug(f"Updated char_to_token_ratio={self.char_to_token_ratio:.2f} (sample={current_ratio:.2f})")
        
        # Update total_tokens - token stats are computed from program state
        state.total_tokens = total_tokens
        
        # If marked for pause, pause now (while in ACTING state)
        if state.marked_for_pause:
            self._clear_mark_and_pause(program_id, state)

    def finalize_request_without_usage(self, program_id: str, state: Program) -> None:
        if (state.status != ProgramStatus.REASONING
                or state.state != ProgramState.ACTIVE):
            return
        if state.reasoning_run_start is not None:
            state.recent_reasoning_durations.append(max(0.0, time.time() - state.reasoning_run_start))
            state.reasoning_run_start = None
        state.status = ProgramStatus.ACTING
        state.acting_since = time.time()
        logger.info("Request of program %s ended without usage info", program_id)
        if state.marked_for_pause:
            self._clear_mark_and_pause(program_id, state)

    def record_cached_tokens(self, program_id: str, state: Program, cached_tokens: int) -> None:
        if program_id in self._first_cached_tokens:
            return
        self._first_cached_tokens[program_id] = cached_tokens
        if cached_tokens <= 0:
            return
        key = state.prefix_group_key
        if cached_tokens > self._group_prefix_len.get(key, 0.0):
            self._group_prefix_len[key] = float(cached_tokens)

    def update_program_tokens_streaming(self, state: Program, delta_tokens: int) -> None:
        """Update program tokens incrementally during streaming.
        
        Called periodically (e.g., every 20 tokens) during streaming to 
        update the token counts in real-time. The program is in REASONING
        state during streaming.
        
        Args:
            state: The program state
            delta_tokens: Number of new tokens since last update
        """
        # Just update program's total_tokens - backend stats are computed from program state
        state.total_tokens += delta_tokens

    async def release_program(self, program_id: str) -> bool:
        """Stop a program and release its resources.
        
        Removes tokens from tracking. Resume of waiting programs is handled by the scheduler.
        """
        if program_id not in self.programs:
            return False
        
        state = self.programs[program_id]
        backend_url = state.backend_url
        backend = self.backends.get(state.backend_url) if state.backend_url else None
        
        # Clean up based on current lifecycle state
        if state.state == ProgramState.PAUSED:
            # Remove from waiting queue
            await self._remove_from_global_waiting_queue(program_id)
            if state.waiting_event:
                state.waiting_event.set()  # Unblock any waiting coroutine
        elif backend and state.state == ProgramState.ACTIVE:
            backend.unregister_program(program_id)
        
        # Clear mark if was marked
        if backend and state.marked_for_pause:
            self._debit_future(backend, state)
            state.marked_for_pause = False

        warm_url = backend_url or state.origin_backend
        if warm_url in self._warm_queues:
            self._warm_queues[warm_url].pop(program_id, None)
        
        state.state = ProgramState.TERMINATED
        
        # Remove from programs dict
        del self.programs[program_id]
        
        logger.info(f"Released and removed program: {program_id}")
        if self.scheduling_enabled and backend_url:
            await self._release_program_on_backend(backend_url, program_id)
        return True

    def get_programs_on_backend(self, backend_url: str) -> Dict[str, Program]:
        """Get all programs assigned to a specific backend."""
        return {
            pid: state for pid, state in self.programs.items()
            if state.backend_url == backend_url
        }

    # -------------------------------------------------------------------------
    # Global Paused Pool
    # -------------------------------------------------------------------------

    def _add_to_global_waiting_queue_sync(
        self, program_id: str, state: Program, backend: Optional[BackendState] = None,
    ) -> None:
        """Add a program to the global waiting queue (synchronous version)."""
        paused_info = PausedInfo(
            program_id=program_id,
            total_tokens=state.total_tokens,
            paused_at=time.time(),
            origin_backend=backend.url if backend else None,
            step_count=state.step_count,
        )
        self.global_waiting_queue[program_id] = paused_info

    async def _remove_from_global_waiting_queue(self, program_id: str) -> Optional[PausedInfo]:
        """Remove a program from the global paused pool."""
        async with self.pause_resume_lock:
            return self.global_waiting_queue.pop(program_id, None)

    def get_paused_counts_by_backend(self) -> Dict[str, int]:
        """Count paused programs per backend using paused pool metadata."""
        counts = {url: 0 for url in self.backends}
        for info in self.global_waiting_queue.values():
            if info.origin_backend in counts:
                counts[info.origin_backend] += 1
        return counts

    # -------------------------------------------------------------------------
    # Capacity-based Scheduling (pause/resume)
    # -------------------------------------------------------------------------

    def _clear_mark_and_pause(self, program_id: str, state: Program) -> None:
        """Clear the mark from a program and pause it.
        
        Called when a marked program's next request arrives.
        """
        backend = self.backends.get(state.backend_url)
        if not backend:
            return

        # Clear the mark and subtract from future_paused_tokens
        state.marked_for_pause = False
        self._debit_future(backend, state)
        
        # Now actually pause the program
        for bk_url, pid, ptype in self._place_off_gpu(program_id, state, backend):
            self._fire_mark(bk_url, pid, ptype)

    # -------------------------------------------------------------------------
    # Periodic Scheduler
    # -------------------------------------------------------------------------

    async def _scheduler_loop(self):
        """Periodic scheduler loop: check thrashing, update shared_tokens, resume."""
        while not self._scheduler_stop:
            try:
                await asyncio.sleep(self._scheduler_interval)
                await self._scheduled_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}", exc_info=True)

    async def _scheduled_check(self):
        """Periodic check: resume waiting programs, enforce capacity, update metrics."""
        t0 = time.time()
        # Step 1: Fetch fresh metrics
        for backend in self.backends.values():
            await backend.fetch_metrics()
            backend.shared_tokens = backend.shared_prefix_tokens(self._group_prefix_len)

        pending: Dict[str, List[Tuple[str, str]]] = {}
        fired = self._pending_mark_tasks
        self._pending_mark_tasks = []

        def queue(marks: List[Tuple[str, str, str]]) -> None:
            for bk_url, pid, ptype in marks:
                pending.setdefault(bk_url, []).append((pid, ptype))

        async with self.pause_resume_lock:
            now = time.time()
            self._reestimate_parked()
            for backend in self.backends.values():
                self._reconcile_future(backend)
            ready = [b for b in self.backends.values() if b.cache_config]
            for backend in ready:
                queue(self._step_idle_displacement(backend, now))
            for backend in ready:
                queue(self._step_admit_warm(backend, now))
            queue(self._step_admit_waiting(now))
            for backend in ready:
                queue(self._step_gpu_overflow(backend, now))
                queue(self._step_cpu_overflow(backend, now))

        tasks = [
            asyncio.create_task(self._batch_program_ops_on_backend(bk_url, marks))
            for bk_url, marks in pending.items()
        ]
        if tasks or fired:
            results = await asyncio.gather(*tasks, *fired, return_exceptions=True)
            logger.info(
                "[MORI] marks: batches=%d marks=%d fired=%d failed=%d",
                len(tasks), sum(len(m) for m in pending.values()), len(fired),
                sum(1 for r in results if isinstance(r, Exception)),
            )
        for backend in self.backends.values():
            logger.info(
                "[MORI] tick backend=%s gpu=%d warm=%d waiting=%d rem=%d shared=%d "
                "cpu_rem=%d elapsed=%.1fms",
                backend.url, backend.active_program_count,
                len(self._warm_queues.get(backend.url, {})), len(self.global_waiting_queue),
                backend.remaining_capacity(), backend.shared_tokens,
                backend.cpu_remaining_capacity(self._warm_tokens(backend.url)),
                (time.time() - t0) * 1000,
            )

    async def _wait_for_resume(self, program_id: str, state: Program) -> None:
        """Wait for a paused program to be resumed.
        
        If timeout (30 min), force resume the program regardless of capacity.
        """
        if state.waiting_event is None:
            return
        await state.waiting_event.wait()

    def _idleness(self, state: Program, now: float) -> float:
        i = state.idleness(now)
        return i if i is not None else 1.0

    @staticmethod
    def _current_cycle_idleness(state: Program, now: float) -> float:
        r = state.recent_reasoning_durations[-1] if state.recent_reasoning_durations else 0.0
        a = max(0.0, now - state.acting_since) if state.acting_since is not None else 0.0
        return a / (r + a) if r + a > 0 else 0.0

    def _clear_demotion_mark(self, state: Program) -> None:
        if not state.marked_for_pause:
            return
        state.marked_for_pause = False
        self._debit_future(self.backends.get(state.backend_url), state)

    def _credit_future(self, backend: Optional[BackendState], state: Program) -> None:
        state.future_credit = state.total_tokens
        if backend:
            backend.future_paused_tokens += state.future_credit

    def _debit_future(self, backend: Optional[BackendState], state: Program) -> None:
        credit = state.future_credit
        state.future_credit = 0
        if backend:
            backend.future_paused_tokens -= credit
            if backend.future_paused_tokens < 0:
                backend.future_paused_tokens = 0

    def _reconcile_future(self, backend: BackendState) -> None:
        live = sum(p.future_credit for p in backend._programs.values() if p.marked_for_pause)
        if live != backend.future_paused_tokens:
            logger.info("[MORI] future_paused reconcile: backend=%s counter=%d live=%d",
                        backend.url, backend.future_paused_tokens, live)
            backend.future_paused_tokens = live

    def _demote_to_warm(self, program_id: str, state: Program) -> None:
        backend_url = state.backend_url
        backend = self.backends.get(backend_url)
        if not backend:
            return
        self._clear_demotion_mark(state)
        backend.unregister_program(program_id)
        self._warm_queues.setdefault(backend_url, {})[program_id] = state.total_tokens
        state.state = ProgramState.PAUSED
        state.origin_backend = backend_url
        if state.waiting_event is None:
            state.waiting_event = asyncio.Event()
        else:
            state.waiting_event.clear()
        logger.info(
            "[MORI] demoted to WARM: program=%s backend=%s tokens=%d status=%s",
            program_id, backend_url, state.total_tokens, state.status.value,
        )

    def _demote_to_waiting(self, program_id: str, state: Program) -> None:
        self._clear_demotion_mark(state)
        backend_url = state.backend_url or state.origin_backend
        if backend_url:
            backend = self.backends.get(backend_url)
            if backend and program_id in backend._programs:
                backend.unregister_program(program_id)
            if backend_url in self._warm_queues:
                self._warm_queues[backend_url].pop(program_id, None)
        backend = self.backends.get(backend_url) if backend_url else None
        self._add_to_global_waiting_queue_sync(program_id, state, backend)
        state.state = ProgramState.PAUSED
        state.origin_backend = backend_url
        state.backend_url = None
        if state.waiting_event is None:
            state.waiting_event = asyncio.Event()
        else:
            state.waiting_event.clear()
        logger.info(
            "[MORI] demoted to WAITING: program=%s origin=%s tokens=%d",
            program_id, state.origin_backend, state.total_tokens,
        )

    def _promote_to_running(self, program_id: str, state: Program, backend: BackendState) -> None:
        backend_url = backend.url
        if backend_url in self._warm_queues:
            self._warm_queues[backend_url].pop(program_id, None)
        self.global_waiting_queue.pop(program_id, None)
        backend.register_program(program_id, state)
        state.backend_url = backend_url
        state.origin_backend = None
        state.state = ProgramState.ACTIVE
        if state.waiting_event:
            state.waiting_event.set()
            state.waiting_event = None
        logger.info(
            "[MORI] promoted to RUNNING: program=%s backend=%s tokens=%d status=%s step=%d",
            program_id, backend_url, state.total_tokens, state.status.value, state.step_count,
        )

    async def _try_promote_on_arrival(self, program_id: str, state: Program) -> bool:
        from ..backend.state import BUFFER_PER_PROGRAM

        async with self.pause_resume_lock:
            if state.waiting_event is None:
                return True
            if state.state != ProgramState.PAUSED:
                return False
            backend_url = state.backend_url
            if not backend_url:
                return False
            warm_queue = self._warm_queues.get(backend_url)
            if warm_queue is None or program_id not in warm_queue:
                return False
            backend = self.backends.get(backend_url)
            if not backend or not backend.cache_config:
                return False
            required = state.total_tokens + BUFFER_PER_PROGRAM
            remaining = backend.remaining_capacity()
            if remaining < required:
                logger.info(
                    "[MORI] arrival promotion deferred: program=%s required=%d remaining=%d",
                    program_id, required, remaining,
                )
                return False
            self._promote_to_running(program_id, state, backend)
        self._fire_mark(backend_url, program_id, PROGRAM_TYPE_INTERACTIVE)
        logger.info(
            "[MORI] arrival promotion: program=%s backend=%s tokens=%d",
            program_id, backend_url, state.total_tokens,
        )
        return True

    def _warm_tokens(self, backend_url: str) -> int:
        return sum(self._warm_queues.get(backend_url, {}).values())

    def _cpu_has_capacity(self, backend: BackendState, extra_tokens: int) -> bool:
        return backend.cpu_remaining_capacity(self._warm_tokens(backend.url)) >= extra_tokens

    def _cpu_overflow(self, backend: BackendState) -> int:
        return max(0, -backend.cpu_remaining_capacity(self._warm_tokens(backend.url)))

    def _cpu_room(self, backend: BackendState) -> int:
        return backend.cpu_remaining_capacity(self._warm_tokens(backend.url))

    def _parked_demand(self, backend_url: str) -> List[int]:
        from ..backend.state import BUFFER_PER_PROGRAM

        pids = list(self._warm_queues.get(backend_url, {})) + list(self.global_waiting_queue)
        return [
            st.total_tokens + BUFFER_PER_PROGRAM
            for st in (self.programs.get(pid) for pid in pids)
            if st is not None and st.status == ProgramStatus.REASONING
        ]

    def _gpu_keep_ok(self, backend: BackendState, tokens: int) -> bool:
        from ..backend.state import BUFFER_PER_PROGRAM

        remaining = backend.remaining_capacity()
        if remaining < 0:
            return False
        freed = remaining + tokens + BUFFER_PER_PROGRAM
        return not any(remaining < need <= freed for need in self._parked_demand(backend.url))

    def _place_off_gpu(
        self, program_id: str, state: Program, backend: BackendState,
    ) -> List[Tuple[str, str, str]]:
        url = backend.url
        tokens = state.total_tokens
        room = self._cpu_room(backend)
        if room >= tokens:
            decision = "WARM"
        elif self._gpu_keep_ok(backend, tokens):
            decision = "KEEP-GPU"
        else:
            decision = "WAITING"
        logger.info("[MORI] lazy demotion: program=%s tokens=%d cpu_room=%d decision=%s",
                    program_id, tokens, room, decision)
        marks: List[Tuple[str, str, str]] = []
        if decision == "WARM":
            self._demote_to_warm(program_id, state)
            marks.append((url, program_id, PROGRAM_TYPE_ASYNC))
        elif decision == "WAITING":
            self._demote_to_waiting(program_id, state)
            marks.append((url, program_id, PROGRAM_TYPE_INACTIVE))
        return marks

    def _reestimate_parked(self) -> None:
        ratio = self.char_to_token_ratio
        if ratio <= 0:
            return
        n = delta = 0
        for st in self.programs.values():
            if (st.state != ProgramState.PAUSED or st.status != ProgramStatus.REASONING
                    or st.reasoning_run_start is not None or st.context_len <= 0):
                continue
            est = int(st.context_len / ratio)
            if est != st.total_tokens:
                delta += est - st.total_tokens
                n += 1
                st.total_tokens = est
        if n:
            logger.info("[MORI] re-estimated %d parked programs (ratio=%.3f, delta=%d)", n, ratio, delta)

    def _parked_returner_fits(self) -> bool:
        from ..backend.state import BUFFER_PER_PROGRAM

        for url, warm_queue in self._warm_queues.items():
            backend = self.backends.get(url)
            if backend is None or not backend.cache_config:
                continue
            remaining = backend.remaining_capacity()
            for pid in warm_queue:
                st = self.programs.get(pid)
                if (st is not None and st.status == ProgramStatus.REASONING
                        and st.total_tokens + BUFFER_PER_PROGRAM <= remaining):
                    return True
        rems = [b.remaining_capacity() for b in self.backends.values() if b.cache_config]
        if not rems:
            return False
        best = max(rems)
        for pid in self.global_waiting_queue:
            st = self.programs.get(pid)
            if (st is not None and st.status == ProgramStatus.REASONING
                    and st.step_count > 1
                    and st.total_tokens + BUFFER_PER_PROGRAM <= best):
                return True
        return False

    def _step_idle_displacement(self, backend: BackendState, now: float) -> List[Tuple[str, str, str]]:
        from ..backend.state import BUFFER_PER_PROGRAM

        marks: List[Tuple[str, str, str]] = []
        pending = [self.programs.get(pid) for pid in self._warm_queues.get(backend.url, {})]
        pending += [self.programs.get(pid) for pid in self.global_waiting_queue]
        pending = [st for st in pending if st is not None and st.status == ProgramStatus.REASONING]
        if not pending:
            return marks
        need = max(st.total_tokens for st in pending) + BUFFER_PER_PROGRAM

        victims = [
            (pid, st) for pid, st in backend._programs.items()
            if st.status == ProgramStatus.ACTING
            and not st.marked_for_pause
            and st.acting_since is not None
            and self._current_cycle_idleness(st, now) >= 0.5
        ]
        victims.sort(key=lambda x: -self._current_cycle_idleness(x[1], now))
        quota = max(1, int(len(victims) * IDLE_DISPLACEMENT_MAX_FRAC))
        for pid, st in victims:
            if quota <= 0 or backend.remaining_capacity() >= need:
                break
            if backend.cpu_remaining_capacity(self._warm_tokens(backend.url)) < st.total_tokens:
                continue
            self._demote_to_warm(pid, st)
            quota -= 1
            marks.append((backend.url, pid, PROGRAM_TYPE_ASYNC))
            logger.info("[MORI] idle displacement: program=%s tokens=%d (pending need=%d)",
                        pid, st.total_tokens, need)
        return marks

    def _step_admit_warm(self, backend: BackendState, now: float) -> List[Tuple[str, str, str]]:
        from ..backend.state import BUFFER_PER_PROGRAM

        marks: List[Tuple[str, str, str]] = []
        warm_queue = self._warm_queues.get(backend.url, {})
        ready = [
            (pid, self.programs[pid]) for pid in list(warm_queue.keys())
            if pid in self.programs and self.programs[pid].status == ProgramStatus.REASONING
        ]
        ready.sort(key=lambda x: self._idleness(x[1], now))
        for pid, st in ready:
            if backend.remaining_capacity() >= st.total_tokens + BUFFER_PER_PROGRAM:
                self._promote_to_running(pid, st, backend)
                marks.append((backend.url, pid, PROGRAM_TYPE_INTERACTIVE))
        return marks

    def _step_admit_waiting(self, now: float) -> List[Tuple[str, str, str]]:
        from ..backend.state import BUFFER_PER_PROGRAM

        marks: List[Tuple[str, str, str]] = []
        backends = [b for b in self.backends.values() if b.cache_config]
        if not backends:
            return marks
        returning: List[Tuple[str, Program]] = []
        new: List[Tuple[str, Program]] = []
        for pid in list(self.global_waiting_queue.keys()):
            st = self.programs.get(pid)
            if not st or st.status == ProgramStatus.ACTING:
                continue
            (new if st.step_count <= 1 else returning).append((pid, st))
        returning.sort(key=lambda x: self._idleness(x[1], now))
        new.sort(key=lambda x: x[1].total_tokens)
        for pid, st in returning + new:
            best = max(backends, key=lambda b: b.remaining_capacity())
            if best.remaining_capacity() >= st.total_tokens + BUFFER_PER_PROGRAM:
                self._promote_to_running(pid, st, best)
                marks.append((best.url, pid, PROGRAM_TYPE_INTERACTIVE))
        return marks

    def _step_gpu_overflow(self, backend: BackendState, now: float) -> List[Tuple[str, str, str]]:
        marks: List[Tuple[str, str, str]] = []
        while backend.capacity_overflow(include_future_release=True) > 0:
            victim = self._pick_gpu_victim(backend, now)
            if victim is None:
                break
            pid, st, is_reasoning = victim
            if is_reasoning:
                st.marked_for_pause = True
                self._credit_future(backend, st)
                logger.info("[MORI] marked REASONING program %s for demotion (tokens=%d)",
                            pid, st.total_tokens)
                continue
            if self._cpu_has_capacity(backend, st.total_tokens):
                self._demote_to_warm(pid, st)
                marks.append((backend.url, pid, PROGRAM_TYPE_ASYNC))
            else:
                self._demote_to_waiting(pid, st)
                marks.append((backend.url, pid, PROGRAM_TYPE_INACTIVE))
        return marks

    def _pick_gpu_victim(self, backend: BackendState, now: float) -> Optional[Tuple[str, Program, bool]]:
        def key(x: Tuple[str, Program]) -> float:
            return -self._idleness(x[1], now)

        acting = [(pid, st) for pid, st in backend._programs.items()
                  if st.status == ProgramStatus.ACTING]
        acting.sort(key=key)
        if acting:
            return (acting[0][0], acting[0][1], False)
        reasoning = [(pid, st) for pid, st in backend._programs.items()
                     if st.status == ProgramStatus.REASONING and not st.marked_for_pause]
        reasoning.sort(key=key)
        if reasoning:
            return (reasoning[0][0], reasoning[0][1], True)
        return None

    def _step_cpu_overflow(self, backend: BackendState, now: float) -> List[Tuple[str, str, str]]:
        marks: List[Tuple[str, str, str]] = []
        warm_queue = self._warm_queues.get(backend.url, {})
        while self._cpu_overflow(backend) > 0 and warm_queue:
            candidates = [(pid, self.programs[pid]) for pid in list(warm_queue.keys())
                          if pid in self.programs]
            if not candidates:
                break
            candidates.sort(key=lambda x: (x[1].status == ProgramStatus.REASONING,
                                           -self._idleness(x[1], now)))
            pid, st = candidates[0]
            bk_url = st.backend_url or st.origin_backend or backend.url
            self._demote_to_waiting(pid, st)
            marks.append((bk_url, pid, PROGRAM_TYPE_INACTIVE))
        return marks

    def _fire_mark(self, backend_url: str, program_id: str, program_type: str) -> None:
        self._pending_mark_tasks.append(asyncio.create_task(
            self._mark_program_type_on_backend(backend_url, program_id, program_type)
        ))

    async def _mark_program_type_on_backend(
        self, backend_url: str, program_id: str, program_type: str,
    ) -> None:
        url = (f"{backend_url.rstrip('/')}/mark_program_type"
               f"?program_id={program_id}&program_type={program_type}")
        try:
            resp = await self.client.post(url, timeout=10.0)
            if resp.status_code != 200:
                logger.warning("[MORI] mark %s=%s on %s: status=%d body=%s",
                               program_id, program_type, backend_url, resp.status_code, resp.text)
        except Exception as e:
            logger.warning("[MORI] mark %s=%s on %s failed: %s",
                           program_id, program_type, backend_url, e)

    async def _release_program_on_backend(self, backend_url: str, program_id: str) -> None:
        url = f"{backend_url.rstrip('/')}/release_program?program_id={program_id}"
        try:
            resp = await self.client.post(url, timeout=10.0)
            if resp.status_code != 200:
                logger.warning("[MORI] release %s on %s: status=%d body=%s",
                               program_id, backend_url, resp.status_code, resp.text)
        except Exception as e:
            logger.warning("[MORI] release %s on %s failed: %s", program_id, backend_url, e)

    async def _batch_program_ops_on_backend(
        self, backend_url: str, marks: List[Tuple[str, str]],
    ) -> None:
        url = f"{backend_url.rstrip('/')}/batch_program_ops"
        payload = {
            "marks": [{"program_id": pid, "program_type": ptype} for pid, ptype in marks],
            "releases": [],
        }
        try:
            resp = await self.client.post(url, json=payload, timeout=10.0)
            if resp.status_code != 200:
                logger.warning("[MORI] batch_program_ops on %s: status=%d body=%s",
                               backend_url, resp.status_code, resp.text)
        except Exception as e:
            logger.warning("[MORI] batch_program_ops on %s failed: %s", backend_url, e)

    def get_warm_queue_metrics_by_backend(self) -> Dict[str, Dict[str, int]]:
        return {
            url: {
                "warm_queue_count": len(queue),
                "warm_queue_tokens": sum(self.programs[pid].total_tokens
                                         for pid in queue if pid in self.programs),
            }
            for url, queue in self._warm_queues.items()
        }

    def get_program_stats(self) -> Dict[str, Any]:
        """Get statistics about all programs."""
        reasoning = sum(1 for p in self.programs.values() if p.status == ProgramStatus.REASONING)
        acting = sum(1 for p in self.programs.values() if p.status == ProgramStatus.ACTING)
        paused = len(self.global_waiting_queue)
        marked = sum(1 for p in self.programs.values() if p.marked_for_pause)
        
        # Per-backend stats
        paused_counts = self.get_paused_counts_by_backend()
        per_backend = {}
        for url, backend in self.backends.items():
            progs = self.get_programs_on_backend(url)
            per_backend[url] = {
                "total": len(progs),
                "reasoning": sum(1 for p in progs.values() if p.status == ProgramStatus.REASONING),
                "acting": sum(1 for p in progs.values() if p.status == ProgramStatus.ACTING),
                "paused": paused_counts.get(url, 0),
                "marked_for_pause": sum(1 for p in progs.values() if p.marked_for_pause),
                "future_paused_tokens": backend.future_paused_tokens,
            }
        
        return {
            "total": len(self.programs),
            "reasoning": reasoning,
            "acting": acting,
            "paused": paused,
            "marked_for_pause": marked,
            "per_backend": per_backend,
        }

    # -------------------------------------------------------------------------
    # Request Proxying
    # -------------------------------------------------------------------------

    @staticmethod
    def extract_total_tokens(payload: Any) -> Optional[int]:
        """Extract total_tokens from the response."""
        if not isinstance(payload, dict):
            return None
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return None
        if "total_tokens" in usage:
            val = usage.get("total_tokens")
            if isinstance(val, (int, float)) and math.isfinite(val):
                return int(val)
        return None

    async def proxy_request(
        self,
        backend: BackendState,
        payload: Dict[str, Any],
        *,
        program_id: str = "",
        on_usage: Callable[[int, int, int], Awaitable[None]] | None = None,
        on_no_usage: Callable[[], None] | None = None,
        on_first_token: Callable[[], None] | None = None,
        on_token: Callable[[], None] | None = None,
        on_token_progress: Callable[[int], None] | None = None,
    ) -> Response:
        """Proxy request to a specific backend.
        
        Args:
            backend: Target backend
            payload: Request payload
            on_usage: Callback with (total_tokens, prompt_tokens, cached_tokens)
            on_first_token: Callback when first token is received (streaming only)
            on_token: Callback for each token (streaming only)
            on_token_progress: Callback with delta token count at intervals (streaming only)
        """
        url = backend.completions_url
        if self.scheduling_enabled:
            payload = dict(payload)
            payload["program_id"] = program_id
        
        if payload.get("stream"):
            return await forward_streaming_request(
                self.client,
                url,
                payload,
                on_usage=on_usage,
                on_no_usage=on_no_usage,
                on_first_token=on_first_token,
                on_token=on_token,
                on_token_progress=on_token_progress,
                keep_program_id=self.scheduling_enabled,
            )
        else:
            return await forward_non_streaming_request(
                self.client,
                url,
                payload,
                on_usage=on_usage,
                on_no_usage=on_no_usage,
                keep_program_id=self.scheduling_enabled,
            )

    async def proxy_get(self, backend_url: str, path: str) -> Response:
        """Proxy a GET request to a backend."""
        url = f"{backend_url.rstrip('/')}{path}"
        return await forward_get_request(self.client, url)
