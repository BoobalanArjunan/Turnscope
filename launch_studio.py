#!/usr/bin/env python3
"""Launches the interactive Spatial Audio Critic & Acoustic Radar Studio in the default browser.
Serves on a lightweight local HTTP server to guarantee full Web Audio API and audio permissions.
"""
import http.server
import socketserver
import threading
import time
import webbrowser
from pathlib import Path

PORT = 8765
DIRECTORY = Path(__file__).resolve().parent


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

    def log_message(self, format, *args):
        # Silent server logging
        pass


def start_server():
    socketserver.TCPServer.allow_reuse_address = True
    for p in range(PORT, PORT + 20):
        try:
            with socketserver.TCPServer(("", p), Handler) as httpd:
                url = f"http://localhost:{p}/spatial_audio_critic_studio.html"
                print(f"\n========================================================")
                print(f"  TurnScope Spatial Audio Critic Studio is Live!")
                print(f"  URL: {url}")
                print(f"========================================================\n")
                # Open browser automatically after a short delay
                threading.Thread(target=lambda: (time.sleep(0.5), webbrowser.open(url))).start()
                httpd.serve_forever()
                break
        except OSError:
            continue


if __name__ == "__main__":
    start_server()
