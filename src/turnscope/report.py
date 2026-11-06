from __future__ import annotations

import csv
import json
from typing import List, Optional

from .analyze import Analysis


def _f(v: Optional[float], unit: str = " s") -> str:
    return "n/a" if v is None else f"{v:.2f}{unit}"


def text(a: Analysis) -> str:
    L = [f"{a.file}  ({a.duration_s:.1f} s)",
         f"  talk time: agent {a.agent_speech_s:.1f} s, customer {a.customer_speech_s:.1f} s",
         f"  agent response delay ({len(a.responses)} turns): median {_f(a.pct(50))}, p90 {_f(a.pct(90))}, worst {_f(max(a.responses) if a.responses else None)}",
         f"  agent talked over the customer: {a.interrupts} time(s)",
         f"  customer barged in on the agent: {a.barge_ins} time(s), agent did not stop {a.agent_not_yielding} time(s)",
         f"  dead air (3 s or more of silence on both sides): {len(a.dead_air)}" + ("".join(f"\n    {s:.1f}-{e:.1f} s" for s, e in a.dead_air[:10]))]
    for w in a.warnings:
        L.append(f"  WARNING: {w}")
    return "\n".join(L)


def to_dict(a: Analysis) -> dict:
    return {"file": a.file, "duration_s": round(a.duration_s, 3), "agent_speech_s": round(a.agent_speech_s, 3),
            "customer_speech_s": round(a.customer_speech_s, 3), "response_delays_s": [round(x, 3) for x in a.responses],
            "median_response_s": a.pct(50), "p90_response_s": a.pct(90), "agent_interrupts": a.interrupts, "customer_barge_ins": a.barge_ins,
            "agent_not_yielding": a.agent_not_yielding, "dead_air": [[round(s, 2), round(e, 2)] for s, e in a.dead_air],
            "leak_agent_to_customer_db": a.leak_agent_to_customer_db, "leak_customer_to_agent_db": a.leak_customer_to_agent_db,
            "events": [{"kind": e.kind, "at_s": round(e.at_s, 3), "value_s": round(e.value_s, 3), "detail": e.detail} for e in a.events],
            "warnings": a.warnings}


def write_json(path, analyses: List[Analysis]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([to_dict(a) for a in analyses], f, indent=1)


def write_csv(path, analyses: List[Analysis]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "duration_s", "responses", "median_response_s", "p90_response_s", "agent_interrupts", "customer_barge_ins",
                    "agent_not_yielding", "dead_air_count", "warnings"])
        for a in analyses:
            w.writerow([a.file, round(a.duration_s, 1), len(a.responses), "" if a.pct(50) is None else round(a.pct(50), 3),
                        "" if a.pct(90) is None else round(a.pct(90), 3), a.interrupts, a.barge_ins, a.agent_not_yielding, len(a.dead_air),
                        "; ".join(a.warnings)])
