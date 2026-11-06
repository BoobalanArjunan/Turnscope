"""turnscope against the simple approach (a fixed loudness threshold per channel, like ffmpeg's silencedetect), on synthetic calls whose
true timing is known. Reproduces the table in docs/. Needs the test helpers.

    python scripts/synthetic_compare.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _baseline  # noqa: E402
from synth import SR, make_call  # noqa: E402
from turnscope import analyse  # noqa: E402


def baseline_responses(agent, customer, sr):
    from turnscope.analyze import from_segments
    return from_segments(_baseline.segments(agent, sr), _baseline.segments(customer, sr), len(agent) / sr).responses


def random_script(rng, turns=6):
    t, script, truth = 1.0, [], []
    for _ in range(turns):
        d = rng.uniform(1.5, 3.5)
        script.append(("customer", t, t + d))
        gap = rng.uniform(0.3, 2.0)
        d2 = rng.uniform(1.5, 3.5)
        script.append(("agent", t + d + gap, t + d + gap + d2))
        truth.append(gap)
        t += d + gap + d2 + rng.uniform(0.3, 1.0)
    return script, truth


def score(got, truth):
    if len(got) != len(truth):
        return None
    return float(np.mean(np.abs(np.array(got) - np.array(truth))))


def main():
    conds = [("clean", dict()), ("noisy (-40 dB floor)", dict(noise_db=-40.0)), ("very noisy (-32 dB floor)", dict(noise_db=-32.0)),
             ("quiet call (-25 dB)", dict(level_db=-25.0)), ("leakage -25 dB", dict(bleed_db=-25.0)), ("leakage -18 dB, room", dict(bleed_db=-18.0, smear=True)),
             ("noisy + leakage -25 dB", dict(noise_db=-40.0, bleed_db=-25.0))]
    print("| condition | turnscope: calls with every turn found | mean error (s) | simple threshold: calls with every turn found | mean error (s) |")
    print("|---|---|---|---|---|")
    for name, kw in conds:
        ok_t = ok_b = 0
        err_t, err_b = [], []
        N = 20
        for seed in range(N):
            rng = np.random.RandomState(seed)
            script, truth = random_script(rng)
            x = make_call(script, seed=seed, **kw)
            a, c = x[:, 0], x[:, 1]
            got = analyse(a, c, SR).responses
            gb = baseline_responses(a, c, SR)
            st, sb = score(got, truth), score(gb, truth)
            if st is not None:
                ok_t += 1
                err_t.append(st)
            if sb is not None:
                ok_b += 1
                err_b.append(sb)
        fmt = lambda e: f"{np.mean(e):.2f}" if e else "-"
        print(f"| {name} | {ok_t}/{N} | {fmt(err_t)} | {ok_b}/{N} | {fmt(err_b)} |")


if __name__ == "__main__":
    main()
