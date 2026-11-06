"""The simple approach turnscope is compared against: a fixed loudness threshold on each channel (like ffmpeg's silencedetect),
with silences shorter than `min_silence_s` bridged. No noise-floor estimate, no leakage handling."""
from turnscope import vad


def segments(x, sr, thr_db=-35.0, min_silence_s=0.3):
    m = vad.frame_db(x, sr) >= thr_db
    gap = int(min_silence_s / vad.HOP_S)
    out, i, n = [], 0, len(m)
    while i < n:
        if m[i]:
            j = i
            while j < n and (m[j] or (m[j:j + gap].any() and j + gap <= n)):
                j += 1
            out.append((i * vad.HOP_S, j * vad.HOP_S))
            i = j
        else:
            i += 1
    return [(s, e) for s, e in out if e - s >= 0.08]
