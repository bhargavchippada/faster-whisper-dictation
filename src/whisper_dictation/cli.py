"""CLI entry point for faster-whisper-dictation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys

from pathlib import Path

from .config import (
    AudioConfig,
    Config,
    CONFIG_DIR,
    CONFIG_FILE,
    EngineConfig,
    HotkeyConfig,
    PID_FILE,
    ServerConfig,
    STATE_FILE,
    VADConfig,
    load_config,
)


def _apply_cli_overrides(
    config: Config,
    *,
    mode: str | None = None,
    hotkey: str | None = None,
    engine: str | None = None,
    server_url: str | None = None,
) -> Config:
    """Apply CLI flag overrides to a config, returning a new Config."""
    hotkey_cfg = config.hotkey
    if mode:
        hotkey_cfg = HotkeyConfig(binding=hotkey_cfg.binding, mode=mode)
    if hotkey:
        hotkey_cfg = HotkeyConfig(binding=hotkey, mode=hotkey_cfg.mode)

    engine_cfg = config.engine
    if engine:
        engine_cfg = EngineConfig(
            type=engine,
            compute_type=engine_cfg.compute_type,
            device=engine_cfg.device,
        )

    server_cfg = config.server
    if server_url:
        server_cfg = ServerConfig(
            url=server_url,
            model=server_cfg.model,
            language=server_cfg.language,
            timeout=server_cfg.timeout,
        )

    return Config(
        server=server_cfg,
        hotkey=hotkey_cfg,
        vad=config.vad,
        audio=config.audio,
        engine=engine_cfg,
    )


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _write_pid() -> None:
    """Write current PID to file with an exclusive lock to prevent races."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl

        fd = os.open(str(PID_FILE), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            print("Daemon already running (lock held).", file=sys.stderr)
            sys.exit(1)
        os.write(fd, str(os.getpid()).encode())
        os.fsync(fd)
        # Keep fd open to hold the lock for the daemon's lifetime
        _write_pid._lock_fd = fd
    except ImportError:
        # fcntl not available (Windows) — fall back to simple write
        PID_FILE.write_text(str(os.getpid()))


def _read_pid() -> int | None:
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)
            return pid
        except (ValueError, ProcessLookupError, PermissionError):
            PID_FILE.unlink(missing_ok=True)
    return None


def _cleanup_pid() -> None:
    # Release the lock fd if held
    fd = getattr(_write_pid, "_lock_fd", None)
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass
        _write_pid._lock_fd = None
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
    config = _apply_cli_overrides(
        config,
        mode=args.mode,
        hotkey=args.hotkey,
        engine=args.engine,
        server_url=args.server_url,
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


_DEFAULT_CONFIG_TEMPLATE = """\
# faster-whisper-dictation configuration
# See: https://github.com/bhargavchippada/faster-whisper-dictation

[server]
url = "http://localhost:10300"        # Whisper server URL
model = "Systran/faster-whisper-large-v3"
language = "en"                       # Language code (e.g. "en", "es", "de")
timeout = 10                          # Request timeout in seconds

[hotkey]
binding = "alt+v"                     # Hotkey combo (e.g. "alt+v", "ctrl+shift+d")
mode = "toggle"                       # "toggle" (press to start/stop) or "hold" (hold to speak)

[vad]
threshold = 0.5                       # Speech detection sensitivity (0.0-1.0, lower = more sensitive)
silence_ms = 800                      # How long to wait after speech stops before transcribing (ms)
min_speech_ms = 250                   # Ignore utterances shorter than this (ms)
max_speech_s = 90.0                   # Maximum single utterance length (seconds)

[audio]
sample_rate = 16000                   # Sample rate in Hz (16000 is optimal for Whisper)
channels = 1                          # Number of audio channels (1 = mono)
# device = "fifine Microphone"        # Uncomment to use a specific mic (see: faster-whisper-dictation devices)

[engine]
type = "server"                       # "server" (Docker/remote) or "local" (built-in faster-whisper)
compute_type = "auto"                 # "auto", "float16" (GPU), "int8" (CPU)
device = "auto"                       # "auto", "cuda", "cpu"
"""


def cmd_config(args) -> None:
    """Show current config or generate a default config file."""
    if args.generate:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        if CONFIG_FILE.exists() and not args.force:
            print(f"Config already exists: {CONFIG_FILE}", file=sys.stderr)
            print("Use --force to overwrite.", file=sys.stderr)
            sys.exit(1)
        CONFIG_FILE.write_text(_DEFAULT_CONFIG_TEMPLATE)
        print(f"Config written to: {CONFIG_FILE}")
        return

    if args.path:
        print(CONFIG_FILE)
        return

    # Show current effective config
    config = load_config()
    print(f"Config file: {CONFIG_FILE} ({'exists' if CONFIG_FILE.exists() else 'not found'})")
    print()
    print(f"[server]")
    print(f"  url          = {config.server.url}")
    print(f"  model        = {config.server.model}")
    print(f"  language     = {config.server.language}")
    print(f"  timeout      = {config.server.timeout}s")
    print()
    print(f"[hotkey]")
    print(f"  binding      = {config.hotkey.binding}")
    print(f"  mode         = {config.hotkey.mode}")
    print()
    print(f"[vad]")
    print(f"  threshold    = {config.vad.threshold}")
    print(f"  silence_ms   = {config.vad.silence_ms}ms")
    print(f"  min_speech   = {config.vad.min_speech_ms}ms")
    print(f"  max_speech   = {config.vad.max_speech_s}s")
    print()
    print(f"[audio]")
    print(f"  sample_rate  = {config.audio.sample_rate}Hz")
    print(f"  channels     = {config.audio.channels}")
    print(f"  device       = {config.audio.device or '(system default)'}")
    print()
    print(f"[engine]")
    print(f"  type         = {config.engine.type}")
    print(f"  compute_type = {config.engine.compute_type}")
    print(f"  device       = {config.engine.device}")


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
    config = _apply_cli_overrides(
        config,
        engine=args.engine,
        server_url=args.server_url,
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
    p_start.add_argument("--config", type=Path, help="config file path")
    p_start.set_defaults(func=cmd_start)

    # stop
    p_stop = sub.add_parser("stop", help="Stop the dictation daemon")
    p_stop.set_defaults(func=cmd_stop)

    # status
    p_status = sub.add_parser("status", help="Show daemon status")
    p_status.set_defaults(func=cmd_status)

    # config
    p_config = sub.add_parser("config", help="Show or generate configuration")
    p_config.add_argument("--generate", action="store_true", help="generate a default config file")
    p_config.add_argument("--force", action="store_true", help="overwrite existing config file")
    p_config.add_argument("--path", action="store_true", help="print config file path only")
    p_config.set_defaults(func=cmd_config)

    # devices
    p_devices = sub.add_parser("devices", help="List audio input devices")
    p_devices.set_defaults(func=cmd_devices)

    # transcribe
    p_trans = sub.add_parser("transcribe", help="Transcribe audio file or recording")
    p_trans.add_argument("file", nargs="?", help="audio file to transcribe")
    p_trans.add_argument("--record", type=float, metavar="SECONDS", help="record N seconds then transcribe")
    p_trans.add_argument("--engine", choices=["server", "local"], help="transcription engine")
    p_trans.add_argument("--server-url", help="whisper server URL")
    p_trans.add_argument("--config", type=Path, help="config file path")
    p_trans.set_defaults(func=cmd_transcribe)

    args = parser.parse_args()
    _setup_logging(args.verbose)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)
