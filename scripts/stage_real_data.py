"""Downloads authentic MagicHub full-duplex conversations (arXiv 2509.04093, CC BY 4.0),
combines dual-mono speaker channels into 2-channel 16 kHz stereo WAVs in recordings/,
and generates labels.csv with ground-truth speech timestamps.
Also reserves unseen conversations in unseen/ for the final check.
"""
from __future__ import annotations

import csv
import io
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
CACHE_RAW = ROOT / "cache_raw"
RECORDINGS_DIR = ROOT / "recordings"
UNSEEN_DIR = ROOT / "unseen"

BASE_URL = "https://huggingface.co/datasets/MagicHub/multi-stream-spontaneous-conversation-training-datasets_english/resolve/main"

# 6 evaluation conversations + 2 held-out unseen conversations
EVAL_CONVERSATIONS = [
    ("conv01.wav", "Group0006_S001_0", "ID164", "ID165"),
    ("conv02.wav", "Group0030_S004_0", "ID024", "ID029"),
    ("conv03.wav", "Group0046_S009_0", "ID023", "ID026"),
    ("conv04.wav", "Group0078_S004_0", "ID098", "ID099"),
    ("conv05.wav", "Group0078_S005_0", "ID098", "ID099"),
    ("conv06.wav", "Group0078_S009_0", "ID098", "ID099"),
]

UNSEEN_CONVERSATIONS = [
    ("unseen01.wav", "Group0078_S013_0", "ID098", "ID099"),
    ("unseen02.wav", "Group0078_S015_0", "ID098", "ID099"),
]

USER_AGENT = "TurnScopeBenchmark/1.0 (https://github.com/BoobalanArjunan/Turnscope; evaluation)"


def download_file(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  [cached] {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [downloading] {dest.name}...", end="", flush=True)
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    dt = time.time() - t0
    mb = dest.stat().st_size / (1024 * 1024)
    print(f" done ({mb:.1f} MB in {dt:.1f}s)")


def parse_transcript(txt_path: Path) -> list[tuple[float, float]]:
    """Extract (start_s, end_s) of primary speech utterances from transcript."""
    segments = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                ts_str, speaker_id, _, text = parts[0], parts[1], parts[2], parts[3]
                if speaker_id != "0":
                    m = re.match(r"\[([0-9.]+),([0-9.]+)\]", ts_str)
                    if m:
                        s, e = float(m.group(1)), float(m.group(2))
                        if e > s:
                            segments.append((s, e))
    return segments


def process_set(conversations, out_dir: Path, labels_csv_path: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_labels = []

    for wav_name, prefix, spk0, spk1 in conversations:
        print(f"\nProcessing {wav_name} ({prefix}):")
        wav0_name = f"{prefix}_{spk0}.wav"
        wav1_name = f"{prefix}_{spk1}.wav"
        txt0_name = f"{prefix}_{spk0}.txt"
        txt1_name = f"{prefix}_{spk1}.txt"

        raw_wav0 = CACHE_RAW / wav0_name
        raw_wav1 = CACHE_RAW / wav1_name
        raw_txt0 = CACHE_RAW / txt0_name
        raw_txt1 = CACHE_RAW / txt1_name

        download_file(f"{BASE_URL}/WAV/{wav0_name}", raw_wav0)
        download_file(f"{BASE_URL}/WAV/{wav1_name}", raw_wav1)
        download_file(f"{BASE_URL}/TXT/{txt0_name}", raw_txt0)
        download_file(f"{BASE_URL}/TXT/{txt1_name}", raw_txt1)

        # Build stereo 2-channel WAV
        dst_wav = out_dir / wav_name
        if not dst_wav.exists() or dst_wav.stat().st_size < 1000:
            print(f"  Combining into 2-channel audio -> {dst_wav.name}...")
            x0, sr0 = sf.read(str(raw_wav0), dtype="float32")
            x1, sr1 = sf.read(str(raw_wav1), dtype="float32")
            if sr0 != sr1:
                raise ValueError(f"Sample rates do not match: {sr0} != {sr1}")
            n = min(len(x0), len(x1))
            stereo = np.column_stack([x0[:n], x1[:n]])
            sf.write(str(dst_wav), stereo, sr0)
            print(f"  Wrote {dst_wav.name} ({n / sr0:.1f}s, {sr0} Hz)")
        else:
            print(f"  [cached stereo] {dst_wav.name}")

        # Extract timestamps
        segs0 = parse_transcript(raw_txt0)
        segs1 = parse_transcript(raw_txt1)
        print(f"  Extracted {len(segs0)} speech segments on ch0 ({spk0}), {len(segs1)} on ch1 ({spk1})")

        for s, e in segs0:
            all_labels.append({"file": wav_name, "channel": 0, "start_s": f"{s:.3f}", "end_s": f"{e:.3f}"})
        for s, e in segs1:
            all_labels.append({"file": wav_name, "channel": 1, "start_s": f"{s:.3f}", "end_s": f"{e:.3f}"})

    print(f"\nWriting labels to {labels_csv_path.name} (total {len(all_labels)} turns)...")
    with open(labels_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "channel", "start_s", "end_s"])
        writer.writeheader()
        writer.writerows(all_labels)
    print(f"Wrote {labels_csv_path}")


def main():
    print("=== Staging MagicHub Full-Duplex Evaluation Dataset ===")
    process_set(EVAL_CONVERSATIONS, RECORDINGS_DIR, ROOT / "labels.csv")

    print("\n=== Staging Unseen Dataset (Reserved for Final Check) ===")
    process_set(UNSEEN_CONVERSATIONS, UNSEEN_DIR, UNSEEN_DIR / "labels.csv")
    print("\nAll datasets staged successfully.")


if __name__ == "__main__":
    main()

