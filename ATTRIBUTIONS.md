# Dataset & Evidence Attributions

## Benchmark Dataset
- **Corpus**: MagicHub Open-Source Full-Duplex Conversational Speech Dataset
- **Paper**: *"Open-Source Full-Duplex Conversational Datasets for Natural and Interactive Speech Synthesis"*, arXiv:2509.04093 (2025)
- **Authors**: Zhitong Zhou, Qingqing Zhang, Lei Luo, Jiechen Liu, Ruohua Zhou (Magic Data Technology & BUCEA)
- **Licence**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Format**: Dual-channel 16 kHz WAV, one speaker isolated per channel, with expert-aligned timestamped transcripts.

## Benchmark Curation & Split
- **Evaluation Set (`recordings/`)**: 6 full-length two-channel conversations (`conv01.wav` through `conv06.wav`), 3,021 labelled ground-truth speech segments, 449 labelled turn hand-overs.
- **Held-out Unseen Set (`unseen/`)**: 2 full-length two-channel conversations (`unseen01.wav`, `unseen02.wav`), 1,128 labelled speech segments, reserved for post-validation checks per project rules.
- **Audio Privacy**: As per project guidelines, raw audio files remain local on disk and are excluded from Git version control.

