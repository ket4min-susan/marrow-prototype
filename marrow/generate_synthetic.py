#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Marrow Protocol: Synthetic Dataset Generator
Generates realistic fake interaction sessions for behavioral biometric evaluation.
Covers 2 distinct input styles with Owner, Tired Owner, Stranger, and Mimic personas.
"""
import math
import random
from typing import List
from .format import InteractionEvent, SessionRecording


def generate_pointer_trajectory(
    start_t: int,
    start_pos: tuple,
    end_pos: tuple,
    speed_factor: float,
    jitter: float,
    steps: int = 25,
) -> List[InteractionEvent]:
    events = []
    x0, y0 = start_pos
    x1, y1 = end_pos
    dist = math.hypot(x1 - x0, y1 - y0)
    dur_ms = int((dist / max(0.05, speed_factor)) * 1000.0)
    dt = max(10, dur_ms // max(1, steps))

    t = start_t
    # Control point for natural Bezier curve
    ctrl_x = (x0 + x1) / 2.0 + random.uniform(-jitter, jitter)
    ctrl_y = (y0 + y1) / 2.0 + random.uniform(-jitter, jitter)

    for i in range(steps + 1):
        u = i / float(steps)
        # Quadratic Bezier
        bx = (1 - u) ** 2 * x0 + 2 * (1 - u) * u * ctrl_x + u ** 2 * x1
        by = (1 - u) ** 2 * y0 + 2 * (1 - u) * u * ctrl_y + u ** 2 * y1
        bx += random.uniform(-jitter * 0.2, jitter * 0.2)
        by += random.uniform(-jitter * 0.2, jitter * 0.2)
        events.append(InteractionEvent(timestamp_ms=t, x=bx, y=by, click="none"))
        t += dt

    return events


def generate_style_1_standard(subject: str, session_id: str, seed: int) -> SessionRecording:
    """Style 1: Standard mouse + physical keyboard."""
    rng = random.Random(seed)
    events: List[InteractionEvent] = []
    curr_t = 1000

    # Motor parameters per persona
    if subject == "owner":
        speed_factor = 0.45
        jitter = 0.04
        dwell = 95
        iki_mean = 180
        corr_prob = 0.03
    elif subject == "tired_owner":
        speed_factor = 0.38  # Fatigued: slightly slower pointer
        jitter = 0.05
        dwell = 110
        iki_mean = 215  # Modest typing latency increase
        corr_prob = 0.05
    elif subject == "stranger":
        speed_factor = 0.85  # Aggressive fast mouse
        jitter = 0.12  # Jerky movements
        dwell = 55
        iki_mean = 110  # Very fast typist
        corr_prob = 0.01
    elif subject == "mimic":
        speed_factor = 0.42  # Intentionally copies owner speed
        jitter = 0.01  # Too robotic / unnaturally straight
        dwell = 95
        iki_mean = 185
        corr_prob = 0.005  # Almost zero natural slip corrections
    else:
        raise ValueError(f"Unknown subject: {subject}")

    pos = (0.2, 0.3)
    text_corpus = "The system validates behavioral motion signatures across heterogeneous devices."

    for word in text_corpus.split():
        # Pointer movement to target area
        target = (rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9))
        traj = generate_pointer_trajectory(curr_t, pos, target, speed_factor, jitter, steps=15)
        events.extend(traj)
        curr_t = traj[-1].timestamp_ms + int(rng.gauss(80, 20))
        pos = target

        # Click event (mouse down and up)
        click_dur = max(30, int(rng.gauss(dwell, 15)))
        events.append(InteractionEvent(curr_t, pos[0], pos[1], click="down"))
        curr_t += click_dur
        events.append(InteractionEvent(curr_t, pos[0], pos[1], click="up"))
        curr_t += int(rng.gauss(150, 40))

        # Typing the word
        for char in word + " ":
            is_corr = rng.random() < corr_prob
            dt = max(40, int(rng.gauss(iki_mean, 35)))
            events.append(InteractionEvent(curr_t, pos[0], pos[1], click="none", text_char=char, text_source="typed", is_correction=is_corr))
            curr_t += dt

    return SessionRecording(session_id=session_id, input_style="standard_mouse_keyboard", subject_label=subject, events=events)


def generate_style_2_assistive(subject: str, session_id: str, seed: int) -> SessionRecording:
    """Style 2: Slow pointer (lip-operated joystick / head mouse) + dictation software."""
    rng = random.Random(seed)
    events: List[InteractionEvent] = []
    curr_t = 1000

    if subject == "owner":
        speed_factor = 0.12  # Slow, deliberate joystick motion
        jitter = 0.015
        dwell = 210  # Dwell click / long latch
        utterance_pause = 1800
        words_per_burst = 4
    elif subject == "tired_owner":
        speed_factor = 0.09  # Even slower
        jitter = 0.025
        dwell = 250
        utterance_pause = 2400
        words_per_burst = 3
    elif subject == "stranger":
        speed_factor = 0.55  # Uses regular mouse speed (violating assistive device mechanics!)
        jitter = 0.08
        dwell = 70
        utterance_pause = 300  # Physical typing cadence, no dictation pauses
        words_per_burst = 1
    elif subject == "mimic":
        speed_factor = 0.14  # Moves pointer slowly to mimic
        jitter = 0.01
        dwell = 190
        utterance_pause = 900  # Poorly timed dictation simulation
        words_per_burst = 2
    else:
        raise ValueError(f"Unknown subject: {subject}")

    pos = (0.5, 0.5)
    utterances = [
        "confirm submission of the audit report",
        "open the ledger and verify transactions",
        "save changes to disk and commit",
    ]

    for utt in utterances:
        target = (rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7))
        traj = generate_pointer_trajectory(curr_t, pos, target, speed_factor, jitter, steps=25)
        events.extend(traj)
        curr_t = traj[-1].timestamp_ms + int(rng.gauss(120, 30))
        pos = target

        # Assistive latch click
        events.append(InteractionEvent(curr_t, pos[0], pos[1], click="down"))
        curr_t += int(rng.gauss(dwell, 25))
        events.append(InteractionEvent(curr_t, pos[0], pos[1], click="up"))
        curr_t += int(rng.gauss(200, 50))

        # Dictation arrival: speech recognizer commits words in rapid bursts (< 2ms intervals), then pauses
        words = utt.split()
        for idx_w in range(0, len(words), words_per_burst):
            burst_chunk = " ".join(words[idx_w : idx_w + words_per_burst]) + " "
            for ch in burst_chunk:
                # Dictation text arrives simultaneously in bursts
                events.append(InteractionEvent(curr_t, pos[0], pos[1], click="none", text_char=ch, text_source="dictation", is_correction=False))
                curr_t += int(rng.uniform(0, 3))
            curr_t += max(400, int(rng.gauss(utterance_pause, 300)))

    return SessionRecording(session_id=session_id, input_style="assistive_joystick_dictation", subject_label=subject, events=events)
