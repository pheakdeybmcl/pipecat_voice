from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

from loguru import logger


class TurnState(str, Enum):
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"


@dataclass
class TurnManager:
    call_uuid: str
    state: TurnState = TurnState.LISTENING
    turn_id: int = 0
    updated_at: float = 0.0

    def __post_init__(self) -> None:
        self.updated_at = time.time()

    def _set(self, new_state: TurnState, reason: str = "") -> None:
        old = self.state
        self.state = new_state
        self.updated_at = time.time()
        if old != new_state:
            if reason:
                logger.info(
                    "Turn state {} -> {} uuid={} reason={}",
                    old.value,
                    new_state.value,
                    self.call_uuid,
                    reason,
                )
            else:
                logger.info(
                    "Turn state {} -> {} uuid={}",
                    old.value,
                    new_state.value,
                    self.call_uuid,
                )

    def on_listening(self, reason: str = "") -> None:
        self._set(TurnState.LISTENING, reason)

    def on_thinking(self, reason: str = "") -> int:
        self.turn_id += 1
        self._set(TurnState.THINKING, reason)
        return self.turn_id

    def on_speaking(self, reason: str = "") -> None:
        self._set(TurnState.SPEAKING, reason)

    def on_interrupted(self, reason: str = "") -> None:
        self._set(TurnState.INTERRUPTED, reason)

