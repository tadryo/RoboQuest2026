"""
Colab static-server helper for the browser-native RoboQuest viewer.

The viewer itself lives in webapp/index.html and runs MuJoCo WASM, Three.js, and
ONNX Runtime Web inside the browser. This helper only serves the repository
directory from the Colab kernel and embeds it in the notebook output.
"""
from __future__ import annotations

import functools
import http.server
import mimetypes
import socket
import socketserver
import threading
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[2]
_SERVERS: dict[int, socketserver.TCPServer] = {}


class _RoboQuestStaticHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:
        return


def _find_free_port(start: int, attempts: int = 20) -> int:
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No free port in range {start}-{start + attempts - 1}")


def _start_static_server(root: Path, port: int) -> int:
    actual_port = _find_free_port(port)
    if actual_port in _SERVERS:
        return actual_port

    mimetypes.add_type("application/wasm", ".wasm")
    mimetypes.add_type("application/javascript", ".mjs")
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("application/onnx", ".onnx")

    handler = functools.partial(_RoboQuestStaticHandler, directory=str(root))
    server = socketserver.ThreadingTCPServer(("0.0.0.0", actual_port), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _SERVERS[actual_port] = server
    return actual_port


def show_colab_browser_viewer(
    mode: str = "flee",
    port: int = 7860,
    height: int = 760,
    repo_root: Optional[str] = None,
) -> None:
    """Serve and embed the browser-native RoboQuest viewer in Colab.

    Parameters
    ----------
    mode:
        "walk" for manual walking practice, "flee" for the oni tag arena.
    port:
        Preferred kernel port. If busy, the next free port is used.
    height:
        Iframe height in pixels.
    repo_root:
        Repository root. Defaults to the installed package root.
    """
    root = Path(repo_root).resolve() if repo_root else ROOT
    webapp = root / "webapp"
    if not (webapp / "index.html").exists():
        raise FileNotFoundError(f"webapp/index.html not found under {root}")

    missing_vendor = [
        webapp / "vendor" / "mujoco" / "mujoco_wasm.js",
        webapp / "vendor" / "ort" / "ort.min.js",
        webapp / "vendor" / "three" / "three.module.js",
    ]
    missing = [str(p.relative_to(root)) for p in missing_vendor if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Browser viewer vendor assets are missing: "
            + ", ".join(missing)
            + ". Run: python scripts/setup_web_vendor.py"
        )

    actual_port = _start_static_server(root, port)
    path = f"/webapp/index.html?mode={mode}#{mode}"

    try:
        from google.colab import output
        from google.colab.output import eval_js
        from IPython.display import HTML, display as ipy_display

        proxy_url = eval_js(f'google.colab.kernel.proxyPort({actual_port})').rstrip('/')
        viewer_url = f"{proxy_url}{path}"
        print(f"🌐 RoboQuest ビューアー → {viewer_url}")
        print("   ↑ 白い画面が出たら上の URL を新しいタブで開いてください")

        # まず serve_kernel_port_as_iframe を試みる
        try:
            output.serve_kernel_port_as_iframe(actual_port, path=path, height=f"{height}px")
        except TypeError:
            try:
                output.serve_kernel_port_as_iframe(actual_port, path=path)
            except Exception:
                # フォールバック: proxyURL をリンクとして表示
                ipy_display(HTML(
                    f'<a href="{viewer_url}" target="_blank" '
                    f'style="font-size:16px;color:#4080ff;">'
                    f'🔗 ビューアーを新しいタブで開く</a>'
                ))
        except Exception:
            ipy_display(HTML(
                f'<a href="{viewer_url}" target="_blank" '
                f'style="font-size:16px;color:#4080ff;">'
                f'🔗 ビューアーを新しいタブで開く</a>'
            ))
    except ImportError:
        print(f"Open http://localhost:{actual_port}{path}")
