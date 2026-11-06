"""From two channels of one call to timing numbers: response delay, interruptions, barge-ins, dead air."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf

from .vad import speech_segments

UTTER_GAP_S = 0.5          # pauses shorter than this inside one speaker's talk do not end the utterance
MAX_RESPONSE_S = 10.0      # a gap longer than this between turns is not a "response" (the call went quiet) but dead air
DEAD_AIR_S = 3.0           # both sides silent for at least this long, between the first and last speech
MIN_OVERLAP_S = 0.2        # overlaps shorter than this are normal conversational overlap and are not counted
STRONG_LEAK_DB = -15.0     # if one side leaks into the other this loudly, timing near overlaps is unreliable


class CallError(Exception):
    pass


@dataclass
class Event:
    kind: str                  # "response" | "customer_pause" | "agent_interrupts" | "customer_barge_in"
    at_s: float
    value_s: float             # response: gap; interrupts: overlap length; barge_in: how long the agent kept talking
    detail: str = ""


@dataclass
class Analysis:
    file: str
    duration_s: float
    agent_speech_s: float
    customer_speech_s: float
    responses: List[float] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
    dead_air: List[Tuple[float, float]] = field(default_factory=list)
    agent_not_yielding: int = 0
    leak_agent_to_customer_db: Optional[float] = None
    leak_customer_to_agent_db: Optional[float] = None
    warnings: List[str] = field(default_factory=list)

    def pct(self, q: float) -> Optional[float]:
        return float(np.percentile(self.responses, q)) if self.responses else None

    @property
    def interrupts(self) -> int:
        return sum(e.kind == "agent_interrupts" for e in self.events)

    @property
    def barge_ins(self) -> int:
        return sum(e.kind == "customer_barge_in" for e in self.events)


def utterances(segs: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out: List[List[float]] = []
    for s, e in segs:
        if out and s - out[-1][1] < UTTER_GAP_S:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def load_channels(path, agent_channel: int = 0, customer: Optional[Path] = None):
    """(agent samples, customer samples, sample rate). One stereo file, or an agent file plus a --customer file."""
    try:
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as e:  # noqa: BLE001
        raise CallError(f"cannot read {Path(path).name}: {e}") from None
    if customer is not None:
        try:
            y, sr2 = sf.read(str(customer), dtype="float32", always_2d=True)
        except Exception as e:  # noqa: BLE001
            raise CallError(f"cannot read {Path(customer).name}: {e}") from None
        if sr2 != sr:
            raise CallError(f"sample rates differ ({sr} and {sr2})")
        a, c = x.mean(axis=1), y.mean(axis=1)
    else:
        if x.shape[1] != 2:
            raise CallError(f"{Path(path).name} has {x.shape[1]} channel(s); two-channel audio (or an agent file plus --customer) is needed")
        if agent_channel not in (0, 1):
            raise CallError("--agent-channel must be 0 or 1")
        a, c = x[:, agent_channel], x[:, 1 - agent_channel]
    n = min(len(a), len(c))
    return a[:n], c[:n], sr


def analyse(agent: np.ndarray, customer: np.ndarray, sr: int, name: str = "") -> Analysis:
    sa, sc, leak_ac, leak_ca = speech_segments(agent, customer, sr)
    return from_segments(sa, sc, len(agent) / sr, name, leak_ac, leak_ca)


def from_segments(sa, sc, duration_s: float, name: str = "", leak_ac=None, leak_ca=None) -> Analysis:
    """Timing numbers from speech segments [(start_s, end_s)] of the agent and the customer (from turnscope's detector or from labels)."""
    res = Analysis(name, duration_s, sum(e - s for s, e in sa), sum(e - s for s, e in sc),
                   leak_agent_to_customer_db=leak_ac, leak_customer_to_agent_db=leak_ca)
    if not sa:
        res.warnings.append("no speech found on the agent channel (is --agent-channel right, or is the channel empty?)")
    if not sc:
        res.warnings.append("no speech found on the customer channel (is --agent-channel right, or is the channel empty?)")
    for who, leak in (("agent into the customer channel", leak_ac), ("customer into the agent channel", leak_ca)):
        if leak is not None and leak > STRONG_LEAK_DB:
            res.warnings.append(f"strong leakage: the {who} is only {-leak:.0f} dB quieter; timing near overlaps is unreliable")
    ua, uc = utterances(sa), utterances(sc)
    allu = sorted([(s, e, "agent") for s, e in ua] + [(s, e, "customer") for s, e in uc])
    for (xs, xe, xw), (ys, ye, yw) in zip(allu, allu[1:]):
        if xw == yw:
            continue
        if ys >= xe:                                              # a clean hand-over
            gap = ys - xe
            if gap > MAX_RESPONSE_S:
                continue
            if yw == "agent":
                res.responses.append(gap)
                res.events.append(Event("response", xe, gap))
            else:
                res.events.append(Event("customer_pause", xe, gap))
        else:                                                     # y starts while x is still talking
            ov = min(xe, ye) - ys
            if ov < MIN_OVERLAP_S:
                continue
            if yw == "agent":
                res.events.append(Event("agent_interrupts", ys, ov, f"over customer for {ov:.2f} s"))
            else:
                kept = xe - ys
                yielded = xe < ye
                if not yielded:
                    res.agent_not_yielding += 1
                res.events.append(Event("customer_barge_in", ys, kept, "agent stopped" if yielded else "agent talked on"))
    alls = sorted(sa + sc)
    if alls:
        end = alls[0][1]
        for s, e in alls[1:]:
            if s - end >= DEAD_AIR_S:
                res.dead_air.append((end, s))
            end = max(end, e)
    return res


def analyse_file(path, agent_channel: int = 0, customer: Optional[Path] = None) -> Analysis:
    a, c, sr = load_channels(path, agent_channel, customer)
    return analyse(a, c, sr, Path(path).name)
