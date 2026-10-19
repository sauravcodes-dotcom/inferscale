"""Start a long job outside the caller's session so it survives the caller exiting.

A job started as a child of an agent or terminal session is killed with that session's process
tree (the first M4.3 sweep was lost this way). This forks, starts a new session (setsid: new
process group, no controlling terminal), redirects output to a log, and keeps the Mac awake
(caffeinate -is) for the job's lifetime.

  uv run python scripts/detach.py logs/m43-run.log -- make m43
Prints the job's PID; the job's exit code is appended to the log.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if "--" not in argv or argv.index("--") != 1 or len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    log = Path(argv[0]).resolve()
    cmd = argv[2:]
    log.parent.mkdir(parents=True, exist_ok=True)
    pid = os.fork()
    if pid:
        print(pid)
        return 0
    os.setsid()
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    quoted = " ".join("'" + c.replace("'", "'\\''") + "'" for c in cmd)
    os.execvp(
        "/bin/sh",
        ["/bin/sh", "-c", f'caffeinate -is {quoted}; echo "exit=$?"'],
    )
    return 0  # not reached


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
