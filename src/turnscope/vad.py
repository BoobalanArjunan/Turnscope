"""Speech / no-speech per channel from loudness only (no model, no GPU). Deliberately simple and explainable.

A frame is speech when it is clearly above the channel's own noise floor. Gaps shorter than MIN_GAP_S are bridged (the pause between
two words), and bursts shorter than MIN_SPEECH_S are dropped (clicks). Two channels of one call are judged together so that the
agent's voice leaking into the customer's channel (and the reverse) is not mistaken for the other person speaking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

FRAME_S = 0.020
HOP_S = 0.010
MARGIN_DB = 12.0          # speech must be this far above the channel's noise floor (10th percentile of frame loudness)
REL_PEAK_DB = 35.0        # ... and within this far of the channel's loud level (95th percentile): rejects steady hum
ABS_MIN_DB = -62.0        # ... and never below this (dBFS)
MIN_GAP_S = 0.12
MIN_SPEECH_S = 0.08
DOMINANCE_DB = 15.0       # while both channels are loud, the one that is this much quieter (each against its own loud level) is a leak, not speech
DOMINANCE_PAD_FRAMES = 4  # 40 ms
LEAK_ENVELOPE_CORR = 0.5  # a leak is only reported if the quiet channel's loudness follows the loud one's at least this closely
BLEED_MIN_FRAMES = 100    # at least 1 s of loud speech on the source channel is needed to estimate leakage


@dataclass
class Channel:
    db: np.ndarray            # loudness per frame (dBFS)
    threshold_db: float
    mask: np.ndarray          # speech per frame, before the leakage correction


def frame_db(x: np.ndarray, sr: int) -> np.ndarray:
    n, hop = int(FRAME_S * sr), int(HOP_S * sr)
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    c = np.concatenate([[0.0], np.cumsum(np.square(x, dtype=np.float64))])
    starts = np.arange(0, len(x) - n + 1, hop)
    rms = np.sqrt((c[starts + n] - c[starts]) / n)
    return 20.0 * np.log10(rms + 1e-9)


def analyse_channel(x: np.ndarray, sr: int) -> Channel:
    db = frame_db(x, sr)
    floor = float(np.percentile(db, 10))
    loud = float(np.percentile(db, 95))
    thr = max(floor + MARGIN_DB, loud - REL_PEAK_DB, ABS_MIN_DB)
    if loud < ABS_MIN_DB + 6.0:                      # a channel with no real signal at all has no speech
        return Channel(db, thr, np.zeros(len(db), dtype=bool))
    return Channel(db, thr, db >= thr)


def estimate_leakage(src: Channel, dst: Channel) -> Optional[float]:
    """How many dB quieter (negative) the other channel is while `src` talks, but ONLY if that channel's loudness really follows
    `src` (a leak rises and falls with the source; real speech on the other side does not). None if there is no sign of a leak."""
    n = min(len(src.db), len(dst.db))
    loud = src.mask[:n] & (src.db[:n] >= float(np.percentile(src.db, 95)) - 10.0) & (src.db[:n] > dst.db[:n] + 3.0)   # its own voice, not a leak of the other side
    if int(loud.sum()) < BLEED_MIN_FRAMES:
        return None
    s_, d_ = src.db[:n][loud], dst.db[:n][loud]
    if float(np.std(s_)) < 1e-6 or float(np.std(d_)) < 1e-6:
        return None
    if float(np.corrcoef(s_, d_)[0, 1]) < LEAK_ENVELOPE_CORR:
        return None
    return float(np.median(d_ - s_))


def bridge_and_drop(mask: np.ndarray) -> List[Tuple[float, float]]:
    """Frame mask -> [(start_s, end_s)] with short gaps bridged and short bursts dropped."""
    segs: List[List[int]] = []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        segs.append([int(s), int(e)])
    gap = int(round(MIN_GAP_S / HOP_S))
    merged: List[List[int]] = []
    for s, e in segs:
        if merged and s - merged[-1][1] <= gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    keep = int(round(MIN_SPEECH_S / HOP_S))
    half = FRAME_S / 2.0
    return [(s * HOP_S + half - HOP_S / 2, e * HOP_S + half - HOP_S / 2) for s, e in merged if e - s >= keep]


def speech_segments(x_a: np.ndarray, x_b: np.ndarray, sr: int):
    """Returns (segments_a, segments_b, leak_a_to_b_db, leak_b_to_a_db). Each leak is None when no leak could be seen."""
    a, b = analyse_channel(x_a, sr), analyse_channel(x_b, sr)
    n = min(len(a.mask), len(b.mask))
    la, lb = float(np.percentile(a.db, 95)), float(np.percentile(b.db, 95))
    rel_a, rel_b = a.db[:n] - la, b.db[:n] - lb                # each channel's loudness against its own loud level
    both = a.mask[:n] & b.mask[:n]
    k = np.ones(2 * DOMINANCE_PAD_FRAMES + 1, dtype=bool)
    dom_a = np.convolve(both & (rel_a < rel_b - DOMINANCE_DB), k, mode="same")          # a little wider, so the edges of a leak go too
    dom_b = np.convolve(both & (rel_b < rel_a - DOMINANCE_DB), k, mode="same")
    ma = a.mask[:n] & ~dom_a
    mb = b.mask[:n] & ~dom_b
    return bridge_and_drop(ma), bridge_and_drop(mb), estimate_leakage(a, b), estimate_leakage(b, a)
