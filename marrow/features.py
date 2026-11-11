#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Marrow Behavioral Biometrics: Device-Agnostic Feature Extraction
Extracts kinematic and temporal signatures from normalized interaction events.
"""
import math
from typing import Any, Dict, List
from .format import InteractionEvent, SessionRecording


def mean(vals: List[float]) -> float:
    return sum(vals) / max(1, len(vals)) if vals else 0.0


def variance(vals: List[float], m: float = None) -> float:
    if not vals or len(vals) < 2:
        return 0.0
    mu = m if m is not None else mean(vals)
    return sum((x - mu) ** 2 for x in vals) / (len(vals) - 1)


def std_dev(vals: List[float], m: float = None) -> float:
    return math.sqrt(variance(vals, m))


def percentile(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    d0 = s[int(f)] * (c - k)
    d1 = s[int(c)] * (k - f)
    return d0 + d1


class FeatureExtractor:
    """Device-agnostic feature extractor for kinematic and input cadence biometrics."""

    FEATURE_NAMES = [
        "pointer_speed_mean",
        "pointer_speed_p90",
        "pointer_accel_std",
        "trajectory_curvature_mean",
        "click_dwell_mean_ms",
        "text_interval_mean_ms",
        "text_interval_p90_ms",
        "text_burst_ratio",
        "correction_rate",
        "idle_pause_ratio",
    ]

    @classmethod
    def extract_features(cls, session: SessionRecording) -> List[float]:
        events = session.events
        if not events:
            return [0.0] * len(cls.FEATURE_NAMES)

        speeds: List[float] = []
        accels: List[float] = []
        curvatures: List[float] = []
        click_dwells: List[float] = []
        text_intervals: List[float] = []
        corrections = 0
        text_events_count = 0
        zero_interval_count = 0
        total_time_ms = max(1, events[-1].timestamp_ms - events[0].timestamp_ms)
        idle_time_ms = 0

        last_click_down: Dict[str, int] = {}
        last_text_t: int = -1

        for i in range(len(events)):
            curr = events[i]

            # Click dwell time calculation
            if curr.click == "down":
                last_click_down["t"] = curr.timestamp_ms
            elif curr.click == "up" and "t" in last_click_down:
                dwell = curr.timestamp_ms - last_click_down.pop("t")
                if 0 <= dwell <= 5000:
                    click_dwells.append(dwell)

            # Text cadence & burstiness
            if curr.text_char:
                text_events_count += 1
                if curr.is_correction:
                    corrections += 1
                if last_text_t >= 0:
                    dt = curr.timestamp_ms - last_text_t
                    if dt <= 5:
                        zero_interval_count += 1
                    if 0 <= dt <= 10000:
                        text_intervals.append(dt)
                last_text_t = curr.timestamp_ms

            # Kinematic motion features (between consecutive points)
            if i > 0:
                prev = events[i - 1]
                dt = curr.timestamp_ms - prev.timestamp_ms
                if dt > 0:
                    dist = math.hypot(curr.x - prev.x, curr.y - prev.y)
                    speed = dist / (dt / 1000.0)  # fractional units per second
                    speeds.append(speed)

                    if dt > 1500:
                        idle_time_ms += dt

                    if i > 1:
                        prev2 = events[i - 2]
                        dt_prev = prev.timestamp_ms - prev2.timestamp_ms
                        if dt_prev > 0:
                            dist_prev = math.hypot(prev.x - prev2.x, prev.y - prev2.y)
                            speed_prev = dist_prev / (dt_prev / 1000.0)
                            accel = (speed - speed_prev) / (dt / 1000.0)
                            accels.append(abs(accel))

                            # Curvature angle change
                            vx1, vy1 = prev.x - prev2.x, prev.y - prev2.y
                            vx2, vy2 = curr.x - prev.x, curr.y - prev.y
                            dot = vx1 * vx2 + vy1 * vy2
                            mag1 = math.hypot(vx1, vy1)
                            mag2 = math.hypot(vx2, vy2)
                            if mag1 > 1e-5 and mag2 > 1e-5:
                                cos_theta = max(-1.0, min(1.0, dot / (mag1 * mag2)))
                                angle = math.acos(cos_theta)
                                curvatures.append(angle)

        f_speed_mean = mean(speeds)
        f_speed_p90 = percentile(speeds, 0.90)
        f_accel_std = std_dev(accels)
        f_curvature_mean = mean(curvatures)
        f_click_dwell = mean(click_dwells) if click_dwells else 80.0
        f_text_mean = mean(text_intervals) if text_intervals else 200.0
        f_text_p90 = percentile(text_intervals, 0.90) if text_intervals else 400.0
        f_burst_ratio = zero_interval_count / max(1, text_events_count)
        f_correction_rate = corrections / max(1, text_events_count)
        f_idle_ratio = min(1.0, idle_time_ms / total_time_ms)

        return [
            round(f_speed_mean, 5),
            round(f_speed_p90, 5),
            round(f_accel_std, 5),
            round(f_curvature_mean, 5),
            round(f_click_dwell, 3),
            round(f_text_mean, 3),
            round(f_text_p90, 3),
            round(f_burst_ratio, 5),
            round(f_correction_rate, 5),
            round(f_idle_ratio, 5),
        ]
