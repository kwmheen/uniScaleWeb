"""로컬에서 web/ 과 models/ 를 제공하고, 페이지가 꺼지면 같이 종료합니다."""

import threading
import time
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODELS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
}


def ensure_models() -> None:
    folder = Path(__file__).resolve().parent / "models"
    folder.mkdir(parents=True, exist_ok=True)
    for name, url in MODELS.items():
        path = folder / name
        if path.exists() and path.stat().st_size > 0:
            continue
        print(f"모델 받는 중: {name}", flush=True)
        urllib.request.urlretrieve(url, path)

PING_TIMEOUT = 6


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        if self.path.split("?", 1)[0] == "/ping":
            self.server.mark_ping()
            self.send_response(204)
            self.end_headers()
            return
        super().do_GET()

    def do_POST(self):
        if self.path.split("?", 1)[0] == "/shutdown":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write("ok".encode())
            self.server.request_shutdown()
            return
        self.send_error(404)

    extensions_map = {
        **getattr(SimpleHTTPRequestHandler, "extensions_map", {}),
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".css": "text/css",
        ".html": "text/html; charset=utf-8",
        ".task": "application/octet-stream",
        ".wasm": "application/wasm",
    }


class AppServer(ThreadingHTTPServer):
    def __init__(self, address, handler):
        super().__init__(address, handler)
        self.last_ping = None
        self.closing = False
        threading.Thread(target=self._watch, daemon=True).start()

    def mark_ping(self):
        self.last_ping = time.monotonic()

    def request_shutdown(self):
        if self.closing:
            return
        self.closing = True
        threading.Thread(target=self.shutdown, daemon=True).start()

    def _watch(self):
        while not self.closing:
            time.sleep(1)
            last = self.last_ping
            if last is None or self.closing:
                continue
            if time.monotonic() - last > PING_TIMEOUT:
                self.request_shutdown()
                return


if __name__ == "__main__":
    ensure_models()
    server = AppServer(("127.0.0.1", 8765), Handler)
    print("http://127.0.0.1:8765/web/index.html", flush=True)
    print("페이지의 종료를 누르거나 브라우저 창을 닫으면 이 서버도 꺼집니다.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("종료되었습니다.", flush=True)
