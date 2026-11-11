#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Marrow Protocol: OS-Neutral Behavioral Event Format
Specification for normalized recording of human-computer interaction events.
"""
import json
from typing import Any, Dict, List, Optional

VALID_SOURCES = {"typed", "on_screen_keyboard", "dictation"}
VALID_CLICKS = {"none", "down", "up", "click"}


class InteractionEvent:
    """A single normalized interaction event with fractional coordinates and modality labels."""

    def __init__(
        self,
        timestamp_ms: int,
        x: float,
        y: float,
        click: str = "none",
        text_char: str = "",
        text_source: str = "typed",
        is_correction: bool = False,
    ):
        self.timestamp_ms = int(timestamp_ms)
        self.x = float(max(0.0, min(1.0, x)))
        self.y = float(max(0.0, min(1.0, y)))
        self.click = click if click in VALID_CLICKS else "none"
        self.text_char = text_char
        self.text_source = text_source if text_source in VALID_SOURCES else "typed"
        self.is_correction = bool(is_correction)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "t": self.timestamp_ms,
            "x": round(self.x, 5),
            "y": round(self.y, 5),
        }
        if self.click != "none":
            d["c"] = self.click
        if self.text_char:
            d["ch"] = self.text_char
            d["src"] = self.text_source
            if self.is_correction:
                d["corr"] = True
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InteractionEvent":
        return cls(
            timestamp_ms=d.get("t", 0),
            x=d.get("x", 0.0),
            y=d.get("y", 0.0),
            click=d.get("c", "none"),
            text_char=d.get("ch", ""),
            text_source=d.get("src", "typed"),
            is_correction=d.get("corr", False),
        )


class SessionRecording:
    """A complete session recording containing metadata and an ordered sequence of events."""

    def __init__(self, session_id: str, input_style: str, subject_label: str, events: Optional[List[InteractionEvent]] = None):
        self.session_id = session_id
        self.input_style = input_style
        self.subject_label = subject_label
        self.events: List[InteractionEvent] = events or []

    def add_event(self, ev: InteractionEvent) -> None:
        self.events.append(ev)

    def save_jsonl(self, filepath: str) -> None:
        meta = {
            "session_id": self.session_id,
            "input_style": self.input_style,
            "subject_label": self.subject_label,
            "event_count": len(self.events),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
            for ev in self.events:
                f.write(json.dumps(ev.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load_jsonl(cls, filepath: str) -> "SessionRecording":
        events = []
        meta = {}
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if "meta" in d:
                    meta = d["meta"]
                else:
                    events.append(InteractionEvent.from_dict(d))
        return cls(
            session_id=meta.get("session_id", "unknown"),
            input_style=meta.get("input_style", "generic"),
            subject_label=meta.get("subject_label", "unknown"),
            events=events,
        )
