# turnscope

Timing of a two-channel call recording (agent on one channel, customer on the other): how long the agent takes to answer, who interrupts whom, and where dead air happens.

**Status: alpha, 0.1.0a1. Tested on synthetic calls only. Not yet checked on real conversations.** Do not trust the numbers on real calls until you have run `docs/REAL_DATA_CHECK.md`.

## Use

    pip install -e .
    turnscope analyze call.wav                       # channel 0 = agent, channel 1 = customer
    turnscope analyze call.wav --agent-channel 1 --json out.json --csv out.csv
    turnscope analyze call.wav --max-median-response 1.2 --max-p90-response 2.5   # exit code 1 if too slow

It reports: response delay per hand-over (median, p90), agent interruptions of the customer, customer barge-ins (and whether the agent stopped), dead air of 3 s or more, and warnings (for example, one channel bleeding into the other).

## How it works (short)

Energy-based speech detection per channel: noise floor from the quietest 10% of frames, limits relative to the channel's own peak. Short gaps are bridged, short bursts dropped. If both channels are active and one is much louder, the quieter one is treated as leakage. Utterances closer than 0.5 s are merged. No model, no GPU, no network.

## What is proven and what is not

Synthetic calls with known turn timing (`scripts/synthetic_compare.py`, 20 random calls per row, 6 turns each). "Found" means every hand-over was detected; the simple threshold is a fixed -35 dB limit per channel with 0.3 s bridging, like an ffmpeg `silencedetect` script.

| condition | turnscope found | error (s) | simple threshold found | error (s) |
|---|---|---|---|---|
| clean | 20/20 | 0.02 | 20/20 | 0.02 |
| noisy (-40 dB floor) | 20/20 | 0.02 | 20/20 | 0.02 |
| very noisy (-32 dB floor) | 20/20 | 0.03 | 0/20 | - |
| quiet call (-25 dB) | 20/20 | 0.02 | 0/20 | - |
| leakage -25 dB | 20/20 | 0.02 | 20/20 | 0.02 |
| leakage -18 dB, room | 20/20 | 0.02 | 0/20 | - |
| noisy + leakage -25 dB | 20/20 | 0.02 | 0/20 | - |

Read this carefully:

- The generator is mine, and speech is syllable-like noise, not real voices.
- The fixed -35 dB baseline is a weak opponent on quiet or noisy calls; a hand-tuned threshold per file would do better. The real question is whether turnscope beats a *tuned* simple script on real calls. That is untested.
- Leakage handling is validated only on synthetic leakage. Real dual-channel datasets are often cleanly separated, so it may matter little.
- All thresholds (dominance, 3 s dead air, 0.5 s merge) are provisional.

## Stop rule

Run `docs/REAL_DATA_CHECK.md`. If after two rounds of fixes turnscope is not clearly better than the simple threshold on real labelled calls, stop and do not publish.

## Tests

    pip install -e .[dev] && pytest     # 41 tests

Planted-bug (mutation) checks were run on the core logic; no survivors after the last round.

MIT licence.
