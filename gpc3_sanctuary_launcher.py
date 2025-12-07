#!/usr/bin/env python3
"""
GPC3 Sanctuary Launcher (backend + local HTML front-end)
--------------------------------------------------------

This launcher:

- Starts the existing FastAPI backend defined in `gpc3_sanctuary.py`.
- Leaves all API / websocket paths exactly as the backend defines them.
- Opens your existing Sanctuary HTML file from disk via a `file://` URL,
  so the UI behaves like it did on your original desktop setup.
- Prefers Chrome/Chromium/Brave, falls back to the system default browser.
- Shuts down cleanly on Ctrl+C.

Directory layout this expects (what you already have)::

    gpc_santuary/
      ├─ gpc3_sanctuary.py
      ├─ gpc3_sanctuary.html   (3D background version)

Usage (from inside the gpc_santuary folder)::

    python3 gpc3_sanctuary_launcher.py

You can safely rename this file to `gpc3_sanctuary_launcher.py`.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI
    import uvicorn
except ImportError as exc:  # pragma: no cover - environment bootstrap
    print("\n[Sanctuary Launcher] Missing dependencies.")
    print("You need to install FastAPI and Uvicorn in this environment:")
    print("    pip install fastapi uvicorn duckduckgo-search\n")
    raise SystemExit(1) from exc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Preferred front-end HTML file names, in order. First one that exists wins.
FRONTEND_CANDIDATES = [
    "gpc3_sanctuary_bg3d_select.html",
    "gpc3_sanctuary.html",
]


# ---------------------------------------------------------------------------
# Backend loading
# ---------------------------------------------------------------------------

def load_backend_module() -> Any:
    """Dynamically import `gpc3_sanctuary.py` from this directory."""
    here = Path(__file__).resolve().parent
    target = here / "gpc3_sanctuary.py"

    if not target.exists():
        print("\n[Sanctuary Launcher] Could not find 'gpc3_sanctuary.py'")
        print(f"Expected path: {target}")
        raise SystemExit(1)

    spec = importlib.util.spec_from_file_location("gpc3_sanctuary", target)
    if spec is None or spec.loader is None:
        print("\n[Sanctuary Launcher] Failed to create import spec.")
        raise SystemExit(1)

    module = importlib.util.module_from_spec(spec)
    sys.modules["gpc3_sanctuary"] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def create_app() -> FastAPI:
    """Obtain the FastAPI app from the backend in a flexible way.

    We support three patterns, in this order:

    1. A `create_app()` factory on the backend module.
    2. An `app` attribute (already a FastAPI instance).
    3. A legacy `open_sanctuary(app: FastAPI)` hook we can call.
    """
    module = load_backend_module()

    if hasattr(module, "create_app"):
        app = module.create_app()  # type: ignore[assignment]
        print("[Sanctuary Launcher] Using backend.create_app()")
        return app

    if hasattr(module, "app"):
        app_obj = module.app  # type: ignore[attr-defined]
        print("[Sanctuary Launcher] Using backend.app instance")
        if isinstance(app_obj, FastAPI):
            return app_obj
        else:
            print("[Sanctuary Launcher] backend.app exists but is not a FastAPI instance.")

    if hasattr(module, "open_sanctuary"):
        print("[Sanctuary Launcher] Using backend.open_sanctuary(app) hook")
        app = FastAPI()
        module.open_sanctuary(app)  # type: ignore[call-arg]
        return app

    print("\n[Sanctuary Launcher] Could not find a way to create the FastAPI app.")
    print("Expected one of: create_app(), app, or open_sanctuary(app).")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Browser helper
# ---------------------------------------------------------------------------

def _open_browser_when_ready(frontend_path: Path, server_url: str) -> None:
    """Non-blocking browser opener.

    We wait briefly for the backend to start, then open the *local HTML file*
    via a `file://` URL so the UI continues to be served directly from disk,
    while it talks to the backend on localhost (e.g. ws://127.0.0.1:8000/...).
    """
    time.sleep(1.5)

    if not frontend_path.exists():
        print(f"[Sanctuary Launcher] Front-end file missing: {frontend_path}")
        print(f"[Sanctuary Launcher] You can still open {server_url} manually if needed.")
        return

    file_url = frontend_path.resolve().as_uri()
    print(f"[Sanctuary Launcher] Opening Sanctuary UI at {file_url}")

    chrome_candidates = [
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        "brave-browser",
    ]

    for cmd in chrome_candidates:
        if shutil.which(cmd):
            try:
                os.spawnlp(os.P_NOWAIT, cmd, cmd, file_url)
                print(f"[Sanctuary Launcher] Opened {cmd!r} with Sanctuary UI.")
                return
            except OSError:
                continue

    try:
        webbrowser.open(file_url, new=1)
        print("[Sanctuary Launcher] Opened default browser with Sanctuary UI.")
    except Exception as exc:  # pragma: no cover - best-effort UX
        print(f"[Sanctuary Launcher] Failed to open browser automatically: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    host = "127.0.0.1"
    port = 8000
    server_url = f"http://{host}:{port}/"

    here = Path(__file__).resolve().parent
    # Pick the first front-end HTML file that exists.
    frontend: Path | None = None
    for name in FRONTEND_CANDIDATES:
        candidate = here / name
        if candidate.exists():
            frontend = candidate
            break

    if frontend is None:
        print("\n[Sanctuary Launcher] Could not find a Sanctuary HTML front-end.")
        print("Looked for one of:")
        for name in FRONTEND_CANDIDATES:
            print(f"  - {name}")
        print("You can still start the backend, but you'll need to open the UI manually.")
        frontend = here / FRONTEND_CANDIDATES[-1]  # last candidate as placeholder

    app = create_app()

    print("\n[Sanctuary Launcher] GPC3 Sanctuary backend is starting...")
    print(f"[Sanctuary Launcher] Backend URL (for websockets/API): {server_url}")
    print(f"[Sanctuary Launcher] Front-end HTML file: {frontend}")
    print("[Sanctuary Launcher] A browser should open with the UI once the server is ready.\n")

    opener = threading.Thread(
        target=_open_browser_when_ready, args=(frontend, server_url), daemon=True
    )
    opener.start()

    try:
        uvicorn.run(app, host=host, port=port, reload=False, log_level="info")
    except KeyboardInterrupt:
        print("\n[Sanctuary Launcher] Ctrl+C received, shutting down Sanctuary...\n")
    finally:
        print("[Sanctuary Launcher] Goodbye.\n")


if __name__ == "__main__":  # pragma: no cover
    main()
