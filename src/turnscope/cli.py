from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .analyze import CallError, analyse_file
from .report import text, write_csv, write_json

AUDIO_EXT = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="turnscope", description="Timing of a two-channel call: agent response delay, interruptions, dead air.")
    ap.add_argument("--version", action="version", version=f"turnscope {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("analyze", help="analyse a two-channel recording, or every recording in a folder")
    c.add_argument("source", type=Path, help="a stereo file (one speaker per channel) or a folder of them")
    c.add_argument("--agent-channel", type=int, default=0, choices=(0, 1), help="which channel is the agent (default 0 = left)")
    c.add_argument("--customer", type=Path, help="the customer's own mono file, when the agent file is mono")
    c.add_argument("--json", type=Path, help="write all numbers and events to this JSON file")
    c.add_argument("--csv", type=Path, help="write one summary row per call to this CSV file")
    c.add_argument("--max-median-response", type=float, help="exit 1 if any call's median response delay is above this many seconds")
    c.add_argument("--max-p90-response", type=float, help="exit 1 if any call's p90 response delay is above this many seconds")

    s = sub.add_parser("studio", help="launch interactive spatial audio critic & radar studio in browser")
    s.add_argument("--port", type=int, default=8765, help="local port to serve on (default: 8765)")
    a = ap.parse_args(argv)

    if a.cmd == "studio":
        import http.server
        import socketserver
        import threading
        import time
        import webbrowser

        root_dir = Path(__file__).resolve().parent.parent.parent
        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(root_dir), **kwargs)
            def log_message(self, format, *args):
                pass

        socketserver.TCPServer.allow_reuse_address = True
        for p in range(a.port, a.port + 20):
            try:
                with socketserver.TCPServer(("", p), Handler) as httpd:
                    url = f"http://localhost:{p}/spatial_audio_critic_studio.html"
                    print(f"\nTurnScope Spatial Audio Critic Studio running at: {url}\nOpening browser...")
                    threading.Thread(target=lambda: (time.sleep(0.5), webbrowser.open(url))).start()
                    try:
                        httpd.serve_forever()
                    except KeyboardInterrupt:
                        print("\nStudio closed.")
                        return 0
                    break
            except OSError:
                continue
        return 0

    if a.source.is_dir():
        files = sorted(p for p in a.source.rglob("*") if p.suffix.lower() in AUDIO_EXT)
    elif a.source.is_file():
        files = [a.source]
    else:
        print(f"error: {a.source} not found", file=sys.stderr)
        return 2
    if not files:
        print("error: no audio files found", file=sys.stderr)
        return 2
    results, errors = [], 0
    for p in files:
        try:
            results.append(analyse_file(p, a.agent_channel, a.customer))
            print(text(results[-1]))
        except CallError as e:
            errors += 1
            print(f"ERROR {e}", file=sys.stderr)
    if a.json:
        write_json(a.json, results)
    if a.csv:
        write_csv(a.csv, results)
    over = [r.file for r in results
            if (a.max_median_response is not None and r.pct(50) is not None and r.pct(50) > a.max_median_response)
            or (a.max_p90_response is not None and r.pct(90) is not None and r.pct(90) > a.max_p90_response)]
    if over:
        print(f"\nresponse delay above the limit: {', '.join(over)}")
    return 1 if (over or (errors and not results)) else 0


if __name__ == "__main__":
    sys.exit(main())
