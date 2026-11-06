"""Synthetic calls with KNOWN timing: speech-like syllables on two channels, a noise floor, and optional leakage between channels."""
import numpy as np

SR = 16000


def utterance(t0, t1, seed, sr=SR, f0=None):
    """Voiced, syllable-like sound from t0 to t1 seconds (short pauses between syllables, like talking)."""
    rng = np.random.RandomState(seed)
    n = int((t1 - t0) * sr)
    out = np.zeros(n, dtype=np.float64)
    f0 = f0 or rng.uniform(110, 220)
    t = 0.0
    while t < (t1 - t0) - 0.03:
        d = min(rng.uniform(0.14, 0.26), (t1 - t0) - t)
        m = int(d * sr)
        tt = np.arange(m) / sr
        f = f0 * (1 + 0.05 * np.sin(2 * np.pi * 4 * tt + rng.rand() * 6))
        ph = 2 * np.pi * np.cumsum(f) / sr
        s = sum(np.sin(k * ph) / k ** 0.7 for k in range(1, 12))
        env = np.sin(np.pi * np.arange(m) / m) ** 0.5
        i = int(t * sr)
        out[i:i + m] += (0.25 * s * env)[: len(out) - i]
        t += d + rng.uniform(0.03, 0.08)                    # a short pause between syllables
    return out


def make_call(script, seconds=None, sr=SR, noise_db=-58.0, bleed_db=None, seed=0, level_db=0.0, smear=False):
    """script: [(speaker, start_s, end_s)] with speaker 'agent' (channel 0) or 'customer' (channel 1). Returns (n, 2) float32."""
    seconds = seconds or max(e for _, _, e in script) + 1.0
    n = int(seconds * sr)
    ch = {"agent": np.zeros(n), "customer": np.zeros(n)}
    for k, (who, s, e) in enumerate(script):
        u = utterance(s, e, seed * 100 + k, sr)
        i = int(s * sr)
        ch[who][i:i + len(u)] += u[: n - i]
    rng = np.random.RandomState(seed + 999)
    gain = 10 ** (level_db / 20)
    a, c = ch["agent"] * gain, ch["customer"] * gain
    if bleed_db is not None:
        k = 10 ** (bleed_db / 20)
        ca, ag = ch["customer"] * gain, ch["agent"] * gain
        if smear:                                                   # leakage through a room: a delayed, smeared copy rather than a clean scaled one
            h = np.zeros(int(0.02 * sr))
            h[int(0.004 * sr)] = 1.0
            h[int(0.006 * sr):] += 0.35 * np.random.RandomState(5).randn(len(h) - int(0.006 * sr)) * np.exp(-np.arange(len(h) - int(0.006 * sr)) / (0.004 * sr))
            h /= np.sqrt((h ** 2).sum())
            ca, ag = np.convolve(ca, h)[:n], np.convolve(ag, h)[:n]
        a, c = a + k * ca, c + k * ag
    nz = 10 ** (noise_db / 20)
    a = a + nz * rng.randn(n)
    c = c + nz * rng.randn(n)
    return np.stack([a, c], axis=1).astype(np.float32)
