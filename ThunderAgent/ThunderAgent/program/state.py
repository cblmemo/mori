"""Program state management."""
import asyncio
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..profile.state import ProfileState


class ProgramStatus(Enum):
    """What the program is currently doing.
    
    REASONING: On GPU, running inference in vLLM
    ACTING: Off GPU, executing tool or waiting for next request
    """
    REASONING = "reasoning"
    ACTING = "acting"


class ProgramState(Enum):
    """Lifecycle state of a program.
    
    ACTIVE: Program is running normally
    PAUSED: Program is paused (waiting in queue)
    TERMINATED: Program has completed/been released
    """
    ACTIVE = "active"
    PAUSED = "paused"
    TERMINATED = "terminated"


@dataclass
class Program:
    """A single program (task) with its status and state."""
    program_id: str
    backend_url: Optional[str] = None  # Which backend this program is assigned to (None if paused/waiting)
    origin_backend: Optional[str] = None  # Backend URL before pause (for resume fallback)
    status: ProgramStatus = ProgramStatus.ACTING  # What the program is doing (reasoning/acting)
    state: ProgramState = ProgramState.ACTIVE  # Lifecycle state (active/paused/terminated)
    context_len: int = 0
    total_tokens: int = 0
    step_count: int = 0
    profile: Optional["ProfileState"] = None  # Profile timing data (when profiling enabled)
    waiting_event: Optional[asyncio.Event] = field(default=None, repr=False)  # Event to wait on when paused
    marked_for_pause: bool = False  # Mark REASONING program to pause when it becomes ACTING
    future_credit: int = 0
    acting_since: Optional[float] = None  # time.time() when status changed to ACTING
    reasoning_run_start: Optional[float] = None
    recent_reasoning_durations: deque = field(default_factory=lambda: deque(maxlen=5))
    recent_acting_durations: deque = field(default_factory=lambda: deque(maxlen=5))
    prefix_group_key: Optional[str] = None

    def idleness(self, now: float) -> Optional[float]:
        reason = sum(self.recent_reasoning_durations)
        act = sum(self.recent_acting_durations)
        if self.reasoning_run_start is not None:
            reason += max(0.0, now - self.reasoning_run_start)
        if self.status == ProgramStatus.ACTING and self.acting_since is not None:
            act += max(0.0, now - self.acting_since)
        denom = reason + act
        if denom <= 0.0:
            return None
        return act / denom
