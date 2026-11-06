"""Real-data check for turnscope: does it get the timing right on REAL two-channel conversations, and is it better than the simple approach?

Needs real dual-channel recordings (one speaker per channel) WITH hand-made speech times, for example the open full-duplex datasets
(arXiv 2509.04093, CC BY 4.0). Make a labels CSV with one row per speech segment:

    file,channel,start_s,end_s
    conv01.wav,0,1.25,3.80
    conv01.wav,1,4.10,6.00

    python scripts/real_data_check.py --audio-dir recordings/ --labels labels.csv --out real_check

Every conversation is judged with each channel taken as "the agent" in turn (both directions count). Reports, for turnscope and for a fixed
loudness threshold (the simple approach): speech-detection F1, and the error of the response delay on hand-over turns found in both the labels
and the audio. Your files are not changed. Writes <out>/results.md.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _baseline  # noqa: E402
from turnscope.analyze import from_segments  # noqa: E402
from turnscope.vad import HOP_S, speech_segments  # noqa: E402

MATCH_S = 0.75


def mask(segs, n):
    m = np.zeros(n, dtype=bool)
    for s, e in segs:
        m[int(s / HOP_S):int(e / HOP_S)] = True
    return m


def f1(pred, truth):
    tp = int((pred & truth).sum())
    p, r = tp / max(1, int(pred.sum())), tp / max(1, int(truth.sum()))
    return 2 * p * r / max(1e-9, p + r), p, r


def response_errors(gt_events, est_events):
    """Match hand-over turns by where the agent starts; error = estimated gap minus labelled gap."""
    errs, missed = [], 0
    est = [(e.at_s + e.value_s, e.value_s) for e in est_events if e.kind == "response"]
    for g in (e for e in gt_events if e.kind == "response"):
        gs = g.at_s + g.value_s
        near = [(abs(s - gs), v) for s, v in est if abs(s - gs) <= MATCH_S]
        if near:
            errs.append(min(near)[1] - g.value_s)
        else:
            missed += 1
    return errs, missed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("real_check"))
    a = ap.parse_args()
    lab = defaultdict(lambda: {0: [], 1: []})
    with open(a.labels, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            lab[r["file"].strip()][int(r["channel"])].append((float(r["start_s"]), float(r["end_s"])))
    if not lab:
        sys.exit("no labels found")
    stats = {k: {"f1": [], "p": [], "r": [], "err": [], "missed": 0, "gt": 0} for k in ("turnscope", "simple threshold")}
    for name, ch in lab.items():
        p = a.audio_dir / name
        if not p.exists():
            print(f"skip {name}: not found")
            continue
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        if x.shape[1] != 2:
            print(f"skip {name}: not two-channel")
            continue
        dur = len(x) / sr
        n = int(dur / HOP_S)
        est_ts = speech_segments(x[:, 0], x[:, 1], sr)[:2]
        est_bl = (_baseline.segments(x[:, 0], sr), _baseline.segments(x[:, 1], sr))
        for method, est in (("turnscope", est_ts), ("simple threshold", est_bl)):
            for c in (0, 1):
                s = stats[method]
                v, pr, rc = f1(mask(est[c], n), mask(sorted(ch[c]), n))
                s["f1"].append(v), s["p"].append(pr), s["r"].append(rc)
            for agent in (0, 1):
                gt = from_segments(sorted(ch[agent]), sorted(ch[1 - agent]), dur)
                es = from_segments(est[agent], est[1 - agent], dur)
                errs, missed = response_errors(gt.events, es.events)
                stats[method]["err"] += errs
                stats[method]["missed"] += missed
                stats[method]["gt"] += sum(e.kind == "response" for e in gt.events)
        print(f"done {name}", flush=True)
    L = ["# turnscope: real-data check", "", f"{len(lab)} labelled conversations.", "",
         "| method | speech F1 | precision | recall | labelled hand-overs | not found in audio | median abs error (s) | p90 abs error (s) |",
         "|---|---|---|---|---|---|---|---|"]
    for k, s in stats.items():
        e = np.abs(s["err"])
        L.append(f"| {k} | {np.mean(s['f1']):.3f} | {np.mean(s['p']):.3f} | {np.mean(s['r']):.3f} | {s['gt']} | {s['missed']} | "
                 f"{np.median(e):.3f} | {np.percentile(e, 90):.3f} |" if len(e) else f"| {k} | - | - | - | {s['gt']} | {s['missed']} | - | - |")
    L += ["", "How to read it: labelled speech times are usually coarse (they include short pauses), so F1 will not be near 1.0 for either method; "
              "compare the two rows. The response-delay error is the number that matters. turnscope is only worth keeping if it is clearly "
              "better than the simple threshold here (lower error AND fewer hand-overs missed).", ""]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "results.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
