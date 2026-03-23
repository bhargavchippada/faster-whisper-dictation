"""CLI entry point for faster-whisper-dictation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys

from .config import CONFIG_DIR, PID_FILE, STATE_FILE, load_config


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _write_pid() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))


def _read_pid() -> int | None:
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            # Check if process is alive
            os.kill(pid, 0)
            return pid
        except (ValueError, ProcessLookupError, PermissionError):
            PID_FILE.unlink(missing_ok=True)
    return None


def _cleanup_pid() -> None:
    PID_FILE.unlink(missing_ok=True)
    STATE_FILE.unlink(missing_ok=True)


def cmd_start(args) -> None:
    """Start the dictation daemon in the foreground."""
    from .daemon import DictationDaemon

    existing = _read_pid()
    if existing is not None:
        print(f"Daemon already running (PID {existing}). Use 'stop' first.", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.config)

    # CLI overrides
    if args.mode:
        config = config.__class__(
            server=config.server,
            hotkey=config.hotkey.__class__(
                binding=config.hotkey.binding,
                mode=args.mode,
            ),
            vad=config.vad,
            audio=config.audio,
            engine=config.engine,
        )
    if args.hotkey:
        config = config.__class__(
            server=config.server,
            hotkey=config.hotkey.__class__(
                binding=args.hotkey,
                mode=config.hotkey.mode,
            ),
            vad=config.vad,
            audio=config.audio,
            engine=config.engine,
        )
    if args.engine:
        config = config.__class__(
            server=config.server,
            hotkey=config.hotkey,
            vad=config.vad,
            audio=config.audio,
            engine=config.engine.__class__(
                type=args.engine,
                compute_type=config.engine.compute_type,
                device=config.engine.device,
            ),
        )
    if args.server_url:
        config = config.__class__(
            server=config.server.__class__(
                url=args.server_url,
                model=config.server.model,
                language=config.server.language,
                timeout=config.server.timeout,
            ),
            hotkey=config.hotkey,
            vad=config.vad,
            audio=config.audio,
            engine=config.engine,
        )

    _write_pid()

    # Write state for status command
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({
        "mode": config.hotkey.mode,
        "hotkey": config.hotkey.binding,
        "engine": config.engine.type,
        "server_url": config.server.url,
    }))

    daemon = DictationDaemon(config)

    def _shutdown(sig, frame):
        daemon.stop()
        _cleanup_pid()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        daemon.start()
        daemon.wait()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        _cleanup_pid()
        sys.exit(1)
    finally:
        _cleanup_pid()


def cmd_stop(args) -> None:
    """Stop the running dictation daemon."""
    pid = _read_pid()
    if pid is None:
        print("No daemon running.")
        return

    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Stopped daemon (PID {pid})")
    except ProcessLookupError:
        print("Daemon process not found, cleaning up.")
    finally:
        _cleanup_pid()


def cmd_status(args) -> None:
    """Show daemon status."""
    pid = _read_pid()
    if pid is None:
        print("Status: stopped")
        return

    print(f"Status: running (PID {pid})")

    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text())
            print(f"  Mode:    {state.get('mode', '?')}")
            print(f"  Hotkey:  {state.get('hotkey', '?')}")
            print(f"  Engine:  {state.get('engine', '?')}")
            print(f"  Server:  {state.get('server_url', '?')}")
        except Exception:
            pass


def cmd_devices(args) -> None:
    """List available audio input devices."""
    from .audio import list_devices

    devices = list_devices()
    if not devices:
        print("No audio input devices found.")
        return

    print(f"{'Index':<8} {'Name':<50} {'Channels':<10} {'Rate'}")
    print("-" * 80)
    for d in devices:
        print(f"{d['index']:<8} {d['name']:<50} {d['channels']:<10} {d['sample_rate']:.0f}")


def cmd_transcribe(args) -> None:
    """Transcribe an audio file or record and transcribe."""
    import numpy as np

    config = load_config(args.config)

    if args.server_url:
        config = config.__class__(
            server=config.server.__class__(
                url=args.server_url,
                model=config.server.model,
                language=config.server.language,
                timeout=config.server.timeout,
            ),
            hotkey=config.hotkey,
            vad=config.vad,
            audio=config.audio,
            engine=config.engine,
        )

    if args.engine:
        config = config.__class__(
            server=config.server,
            hotkey=config.hotkey,
            vad=config.vad,
            audio=config.audio,
            engine=config.engine.__class__(
                type=args.engine,
                compute_type=config.engine.compute_type,
                device=config.engine.device,
            ),
        )

    from .engine.server import ServerEngine

    if config.engine.type == "local":
        from .engine.local import LocalEngine
        engine = LocalEngine(config.server, config.engine)
    else:
        engine = ServerEngine(config.server)

    if args.file:
        import wave

        with wave.open(args.file, "rb") as w:
            sr = w.getframerate()
            frames = w.readframes(w.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        text = engine.transcribe(audio, sr)
        if text:
            print(text)
        else:
            print("No speech detected.", file=sys.stderr)
            sys.exit(1)

    elif args.record:
        import sounddevice as sd

        duration = args.record
        print(f"Recording {duration}s...", file=sys.stderr)
        audio = sd.rec(
            int(duration * config.audio.sample_rate),
            samplerate=config.audio.sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        print("Transcribing...", file=sys.stderr)

        text = engine.transcribe(audio.flatten(), config.audio.sample_rate)
        if text:
            print(text)
        else:
            print("No speech detected.", file=sys.stderr)
            sys.exit(1)
    else:
        print("Provide --file or --record", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="faster-whisper-dictation",
        description="Real-time speech-to-text dictation powered by faster-whisper",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    sub = parser.add_subparsers(dest="command")

    # start
    p_start = sub.add_parser("start", help="Start the dictation daemon")
    p_start.add_argument("--mode", choices=["toggle", "hold"], help="hotkey mode")
    p_start.add_argument("--hotkey", help="hotkey binding (e.g. 'alt+v', 'ctrl+shift+d')")
    p_start.add_argument("--engine", choices=["server", "local"], help="transcription engine")
    p_start.add_argument("--server-url", help="whisper server URL")
    p_start.add_argument("--config", type=lambda p: __import__("pathlib").Path(p), help="config file path")
    p_start.set_defaults(func=cmd_start)

    # stop
    p_stop = sub.add_parser("stop", help="Stop the dictation daemon")
    p_stop.set_defaults(func=cmd_stop)

    # status
    p_status = sub.add_parser("status", help="Show daemon status")
    p_status.set_defaults(func=cmd_status)

    # devices
    p_devices = sub.add_parser("devices", help="List audio input devices")
    p_devices.set_defaults(func=cmd_devices)

    # transcribe
    p_trans = sub.add_parser("transcribe", help="Transcribe audio file or recording")
    p_trans.add_argument("file", nargs="?", help="audio file to transcribe")
    p_trans.add_argument("--record", type=float, metavar="SECONDS", help="record N seconds then transcribe")
    p_trans.add_argument("--engine", choices=["server", "local"], help="transcription engine")
    p_trans.add_argument("--server-url", help="whisper server URL")
    p_trans.add_argument("--config", type=lambda p: __import__("pathlib").Path(p), help="config file path")
    p_trans.set_defaults(func=cmd_transcribe)

    args = parser.parse_args()
    _setup_logging(args.verbose)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)
