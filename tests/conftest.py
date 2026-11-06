import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from synth import SR, make_call  # noqa: E402


def write(path, x, sr=SR):
    sf.write(str(path), x, sr)
    return path
