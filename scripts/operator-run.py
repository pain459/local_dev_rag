#!/usr/bin/env python3
"""Portable outer deadline; terminate only the supervised command's process group."""

import os
import signal
import subprocess
import sys


def stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=0.2)
    except subprocess.TimeoutExpired:
        pass
    # Always kill remaining descendants, even if the group leader already exited.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def main():
    if len(sys.argv) < 3 or not sys.argv[1].isdecimal() or not 1 <= int(sys.argv[1]) <= 86400:
        print("FAIL: timeout must be whole seconds between 1 and 86400.", file=sys.stderr)
        return 2
    seconds, command = int(sys.argv[1]), sys.argv[2:]
    try:
        process = subprocess.Popen(command, start_new_session=True)
    except OSError:
        print(
            "FAIL: operator executable unavailable; check the selected tool override.",
            file=sys.stderr,
        )
        return 127

    def interrupted(signum, frame):
        stop_group(process)
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    try:
        returncode = process.wait(timeout=seconds)
        return returncode if returncode >= 0 else 128 - returncode
    except subprocess.TimeoutExpired:
        print(
            f"FAIL: operator timeout after {seconds}s; command and child processes terminated. "
            "Inspect the service/tool, or deliberately increase its timeout override.",
            file=sys.stderr,
        )
        return 124
    finally:
        stop_group(process)


if __name__ == "__main__":
    sys.exit(main())
