from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .nlu_entities import EntityResolution


@dataclass
class CallMemory:
    max_turns: int = 4
    recent_user_turns: deque[str] = field(default_factory=lambda: deque(maxlen=4))
    service: Optional[str] = None
    service_confidence: float = 0.0
    country: Optional[str] = None
    country_confidence: float = 0.0
    intent: Optional[str] = None
    intent_confidence: float = 0.0
    updated_at: float = 0.0

    def __post_init__(self) -> None:
        # Rebuild deque with configured capacity.
        self.recent_user_turns = deque(maxlen=max(1, self.max_turns))
        self.updated_at = time.time()

    def add_user_turn(self, text: str) -> None:
        t = (text or "").strip()
        if not t:
            return
        self.recent_user_turns.append(t)
        self.updated_at = time.time()

    def apply_context(self, resolution: EntityResolution) -> EntityResolution:
        # Fill missing fields from recent context (same call only).
        if resolution.service is None and self.service and self.service_confidence >= 0.8:
            resolution.service = self.service
            resolution.service_confidence = max(resolution.service_confidence, self.service_confidence * 0.9)
            resolution.inferred_from_context = True
            resolution.debug_notes.append("service_from_context")
        if resolution.country is None and self.country and self.country_confidence >= 0.8:
            resolution.country = self.country
            resolution.country_confidence = max(resolution.country_confidence, self.country_confidence * 0.9)
            resolution.inferred_from_context = True
            resolution.debug_notes.append("country_from_context")
        if resolution.intent is None and self.intent and self.intent_confidence >= 0.8:
            resolution.intent = self.intent
            resolution.intent_confidence = max(resolution.intent_confidence, self.intent_confidence * 0.9)
            resolution.inferred_from_context = True
            resolution.debug_notes.append("intent_from_context")
        return resolution

    def update_from_resolution(self, resolution: EntityResolution) -> None:
        if resolution.service and resolution.service_confidence >= 0.6:
            if resolution.service_confidence >= self.service_confidence or self.service != resolution.service:
                self.service = resolution.service
                self.service_confidence = resolution.service_confidence
        if resolution.country and resolution.country_confidence >= 0.6:
            if resolution.country_confidence >= self.country_confidence or self.country != resolution.country:
                self.country = resolution.country
                self.country_confidence = resolution.country_confidence
        if resolution.intent and resolution.intent_confidence >= 0.6:
            if resolution.intent_confidence >= self.intent_confidence or self.intent != resolution.intent:
                self.intent = resolution.intent
                self.intent_confidence = resolution.intent_confidence
        self.updated_at = time.time()

    def summary(self) -> str:
        parts: list[str] = []
        if self.service:
            parts.append(f"service={self.service}")
        if self.country:
            parts.append(f"country={self.country}")
        if self.intent:
            parts.append(f"intent={self.intent}")
        if not parts:
            return ""
        return ", ".join(parts)

