"""`nde gui`: open the story builder in a local web page."""

import argparse
import contextlib
import os
import socket
import sys
import threading
import webbrowser
from pathlib import Path

from nde.continuity.cli import DEFAULT_CONFIG
from nde.gui.studio import DEFAULT_OWN_BRIEFS
from nde.llm.ledger import DEFAULT_LEDGER_PATH
from nde.runs.config import ConfigError
from nde.runs.runner import RunOptions

HOST = "127.0.0.1"
DEFAULT_PORT = 8771
EXIT_OK = 0
EXIT_USAGE = 2
_BROWSER_DELAY_S = 0.5


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--no-browser", action="store_true", help="print the address without opening a browser"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"the models and budget ({DEFAULT_CONFIG.as_posix()})",
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("runs"), help="where stories are saved (./runs)"
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path(os.environ.get("NDE_LEDGER", DEFAULT_LEDGER_PATH)),
        help="cost ledger with the daily totals (env NDE_LEDGER, default .nde/ledger.sqlite)",
    )
    parser.add_argument(
        "--ideas-dir",
        type=Path,
        default=DEFAULT_OWN_BRIEFS,
        help=f"where typed ideas and examples are saved ({DEFAULT_OWN_BRIEFS.as_posix()})",
    )


def run(args: argparse.Namespace) -> int:
    # Imported here so other commands do not pay for loading the web stack.
    import uvicorn  # noqa: PLC0415

    from nde.gui.app import create_app  # noqa: PLC0415
    from nde.gui.studio import Studio  # noqa: PLC0415

    options = RunOptions(runs_dir=args.runs_dir, environ=dict(os.environ), ledger_path=args.ledger)
    studio = Studio(args.config, options, args.ideas_dir)
    try:
        studio.examples()
    except ConfigError as error:
        print(f"nde gui: {error}", file=sys.stderr)
        return EXIT_USAGE
    listener = socket.socket()
    try:
        listener.bind((HOST, args.port))
    except OSError:
        listener.close()
        print(
            f"nde gui: port {args.port} is in use; Story Builder may already be open at "
            f"http://{HOST}:{args.port}, or choose another with --port",
            file=sys.stderr,
        )
        return EXIT_USAGE
    url = f"http://{HOST}:{args.port}"
    print(f"Story Builder is at {url}\nKeep this window open. Press Ctrl+C here to stop.")
    if not args.no_browser:
        threading.Timer(_BROWSER_DELAY_S, webbrowser.open, [url]).start()
    server = uvicorn.Server(uvicorn.Config(create_app(studio), log_level="warning"))
    # After shutting down, uvicorn raises the interrupt again for its caller to handle.
    with contextlib.suppress(KeyboardInterrupt):
        server.run(sockets=[listener])
    unfinished = " The story being written was not finished." if studio.current else ""
    print(f"Story Builder has stopped.{unfinished}")
    return EXIT_OK
