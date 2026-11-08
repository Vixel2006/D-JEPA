"""Remote entry point used by ``scripts/colab_run.sh``.

Assumes the repository has been unpacked at ``/content/pidjepa`` and the
package has already been installed.  Forwards ``DJEPA_ARGS`` to the CLI and
streams its output live to the Colab cell while also writing it to a log, then
tars the output directory for download.

On failure the log tail is echoed again, so ``colab exec`` surfaces real
errors instead of only a CalledProcessError.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tarfile

REPO = "/content/pidjepa"
OUTPUT_ARCHIVE = "/content/pidjepa_outputs.tar.gz"
LOG_PATH = "/content/pidjepa_run.log"


def _tail(path: str, limit: int) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", errors="replace") as handle:
        text = handle.read()
    if len(text) <= limit:
        return text
    return "... (truncated) ...\n" + text[-limit:]


def _archive_outputs() -> None:
    output_dir = os.path.join(REPO, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(LOG_PATH):
        try:
            import shutil

            shutil.copy2(LOG_PATH, os.path.join(output_dir, "run.log"))
        except OSError as exc:  # pragma: no cover - filesystem dependent
            print(f"[colab] could not copy log: {exc}", flush=True)
    with tarfile.open(OUTPUT_ARCHIVE, "w:gz") as archive:
        archive.add(output_dir, arcname="outputs")
    print(f"[colab] wrote {OUTPUT_ARCHIVE}", flush=True)


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
    arguments = shlex.split(os.environ.get("DJEPA_ARGS", ""))
    command = [sys.executable, "-u", "-m", "djepa", *arguments]
    print("[colab] running:", " ".join(command), flush=True)

    returncode = _stream(command)

    print(f"[colab] exit code: {returncode}", flush=True)
    if returncode != 0:
        print("[colab] --- failure log tail ---", flush=True)
        print(_tail(LOG_PATH, 20_000), flush=True)
        print("[colab] --- end failure log ---", flush=True)

    _archive_outputs()
    return returncode


if __name__ == "__main__":
    # Avoid ``SystemExit(0)``: IPython prints a spurious traceback for it.
    exit_code = main()
    if exit_code:
        raise SystemExit(exit_code)
