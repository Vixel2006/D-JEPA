"""Remote entry point used by ``scripts/colab_run.sh``.

Assumes the repository has been unpacked at ``/content/pidjepa`` and the
dependencies have already been installed.  Forwards ``PIDJEPA_ARGS`` to
``python main.py``, streams its output live to the
Colab cell while also writing it to a log, then tars the run's output directory
for download.

The output directory is discovered from ``--output-dir``, ``--set
run.output_dir=...``, the config file's ``run.output_dir``, or the newest
``report.json`` produced under the repository.  This keeps arbitrary config
paths working without hard-coding ``results/``.

On failure the log tail is echoed again, so ``colab exec`` surfaces real
errors instead of only a CalledProcessError.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path("/content/pidjepa")
OUTPUT_ARCHIVE = Path("/content/pidjepa_outputs.tar.gz")
LOG_PATH = Path("/content/pidjepa_run.log")
DEFAULT_OUTPUT = "runs/djepa"


def _tail(path: Path, limit: int) -> str:
    if not path.exists():
        return ""
    text = path.read_text(errors="replace")
    if len(text) <= limit:
        return text
    return "... (truncated) ...\n" + text[-limit:]


def _flag_value(arguments: list[str], flag: str) -> str | None:
    for index, argument in enumerate(arguments):
        if argument == flag and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith(f"{flag}="):
            return argument.split("=", 1)[1]
    return None


def _override_value(arguments: list[str], key: str) -> str | None:
    for index, argument in enumerate(arguments):
        value = None
        if argument == "--set" and index + 1 < len(arguments):
            value = arguments[index + 1]
        elif argument.startswith("--set="):
            value = argument.split("=", 1)[1]
        if value and value.startswith(f"{key}="):
            return value.split("=", 1)[1]
    return None


def _config_output_dir(arguments: list[str]) -> str | None:
    config_path = _flag_value(arguments, "--config")
    if not config_path:
        return None
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO / path
    if not path.exists():
        return None
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        else:
            values = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ImportError):
        return None
    run_values = values.get("run") if isinstance(values, dict) else None
    if isinstance(run_values, dict):
        output_dir = run_values.get("output_dir")
        return str(output_dir) if output_dir else None
    return None


def _newest_report_directory() -> Path | None:
    newest: tuple[float, Path] | None = None
    for root, directories, files in os.walk(REPO):
        directories[:] = [
            name for name in directories if name not in {".git", "checkpoints", "__pycache__"}
        ]
        if "report.json" not in files:
            continue
        report = Path(root) / "report.json"
        try:
            modified = report.stat().st_mtime
        except OSError:
            continue
        if newest is None or modified > newest[0]:
            newest = (modified, Path(root))
    return newest[1] if newest else None


def resolve_output_dir(arguments: list[str]) -> Path:
    """Find the directory containing the artifacts that should be archived."""
    explicit = _flag_value(arguments, "--output-dir") or _override_value(
        arguments, "run.output_dir"
    )
    if explicit:
        path = Path(explicit).expanduser()
    else:
        configured = _config_output_dir(arguments)
        if configured:
            path = Path(configured).expanduser()
        else:
            path = _newest_report_directory() or Path(DEFAULT_OUTPUT)
    if not path.is_absolute():
        path = REPO / path
    return path.resolve()


def _archive_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if LOG_PATH.exists():
        try:
            import shutil

            shutil.copy2(LOG_PATH, output_dir / "run.log")
        except OSError as exc:  # pragma: no cover - filesystem dependent
            print(f"[colab] could not copy log: {exc}", flush=True)
    archive_name = output_dir.name or "results"
    with tarfile.open(OUTPUT_ARCHIVE, "w:gz") as archive:
        archive.add(output_dir, arcname=archive_name)
    print(f"[colab] wrote {OUTPUT_ARCHIVE} ({archive_name}/)", flush=True)


def _stream(command: list[str]) -> int:
    """Run ``command``, forwarding each line to the cell and the log."""

    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    with open(LOG_PATH, "w", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        return process.wait()


def main() -> int:
    arguments = shlex.split(os.environ.get("PIDJEPA_ARGS", ""))
    command = [sys.executable, "-u", "main.py", *arguments]
    output_dir = resolve_output_dir(arguments)
    print("[colab] running:", " ".join(command), flush=True)
    print(f"[colab] output directory: {output_dir}", flush=True)

    returncode = _stream(command)

    if returncode != 0:
        print(_tail(LOG_PATH, 12000), flush=True)
        raise SystemExit(f"benchmark failed with exit code {returncode}")

    _archive_outputs(output_dir)
    return 0


if __name__ == "__main__":
    # Avoid ``SystemExit(0)``: IPython prints a spurious traceback for it.
    exit_code = main()
    if exit_code:
        raise SystemExit(exit_code)
