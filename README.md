# turnscope

Timing of a two-channel call recording (agent on one channel, customer on the other): how long the agent takes to answer, who interrupts whom, where dead air happens, and intra-turn filler stalls.

**Status: 0.2.0.** Validated on synthetic calls and authentic multi-channel conversational recordings (MagicHub full-duplex corpus, arXiv:2509.04093, CC BY 4.0).

## Use

    pip install turnscope
    turnscope analyze call.wav                       # channel 0 = agent, channel 1 = customer
    turnscope analyze call.wav --agent-channel 1 --json out.json --csv out.csv
    turnscope analyze call.wav --max-median-response 1.2 --max-p90-response 2.5   # exit code 1 if too slow
    turnscope analyze call.wav --stall-gap 0.8 --filler-burst 1.2 --max-median-ttra 2.0

It reports:
- **Response Delay / Time-to-Acknowledge (TTA):** Initial latency per hand-over (median, p90, worst).
- **Intra-Turn Filler Stalls & Time-to-Real-Answer (TTRA):** Detects when an AI voice agent masks generation delay with filler phrases (*"Let me check that..."*) followed by an internal stall before the real response.
- **Agent Interruptions & Customer Barge-Ins:** Who talked over whom, and whether the agent stopped or kept talking.
- **Dead Air:** 3 s or more of silence on both sides.
- **Channel Leakage:** Bleed between channels estimated and warned.

## Intra-Turn Filler Stalls & Dual-Metric Latency (v0.2.0)

Modern LLM voice agents frequently emit an instant acknowledgment burst (*"Sure, let me look into that"*, $\le 1.2\text{ s}$) and then stall for $1\text{–}3\text{ s}$ while retrieving data or awaiting completion. Traditional latency benchmarks only measure the initial voice onset, producing artificially low response times.

TurnScope v0.2.0 introduces **dual-metric conversational latency**:
1. **Time-to-Acknowledge (TTA):** Delay from customer end of speech to the onset of any agent speech (including fillers).
2. **Time-to-Real-Answer (TTRA):** Delay from customer end of speech to the substantive answer after the internal stall.
3. **Filler Stall Detection:** Triggered when an initial burst $\le \text{filler-burst}$ (default `1.2 s`) is followed by silence $\ge \text{stall-gap}$ (default `0.8 s`).

*Acknowledgment: Feature requested and reviewed by AI/ML engineer Praneeth P K via open-source community feedback.*

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

## Real-Data Check Benchmark (MagicHub Full-Duplex Corpus)

Evaluated on authentic dual-track human conversations from the MagicHub full-duplex corpus (arXiv:2509.04093, CC BY 4.0; dual-track 16 kHz WAV) totaling 6 multi-turn conversations and 449 labelled turn transitions (3,021 ground-truth speech segments) using `scripts/real_data_check.py`:

| Method | Speech F1 | Precision | Recall | Labelled Hand-Overs | Not Found in Audio (Missed) | Median Abs Error (s) | P90 Abs Error (s) |
|---|---|---|---|---|---|---|---|
| **TurnScope** | **0.897** | 0.944 | **0.856** | 449 | **126** | **0.277 s** | **0.440 s** |
| Simple Threshold | 0.700 | **0.965** | 0.600 | 449 | 207 | 0.410 s | 1.339 s |

### Key Findings
1. **Accurate Speech Detection**: TurnScope achieves an F1 of **0.897** (vs 0.700 for the simple threshold), driven by a significantly higher speech recall (**85.6% vs 60.0%**), preventing soft speech from being clipped.
2. **Superior Turn Discovery**: TurnScope misses **39% fewer hand-overs** (126 missed vs 207 for simple threshold), tracking rapid conversational hand-offs that a static threshold loses.
3. **Low Response-Delay Error**: Response delay median error is **0.277 s** (32% lower error than baseline's 0.410 s), and P90 tail error is **0.440 s vs 1.339 s** (a **67% error reduction**), providing reliable conversational response timing without GPU or cloud dependency.

## Tests

    pip install -e .[dev] && pytest     # 41 tests

Planted-bug (mutation) checks were run on the core logic; no survivors after the last round.

## License

MIT. Copyright (c) 2026 Boobalan Arjunan.
