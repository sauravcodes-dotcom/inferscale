"""Clean-environment preflight for the final test run (and any measurement run).

Fails (exit 2, printing every reason) unless: on AC power, swap used <= 64 MB, no
Chrome/Safari/Ollama processes, and ports 8000/8081 are free (no stale gateway/engine).

Usage: uv run python scripts/preflight.py [--max-swap-mb 64]
"""

from __future__ import annotations

import argparse
import re
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.memory_sampler import swap_used_mb
from bench.run_experiment import power_source, running_process_names

FORBIDDEN = (r"^Google Chrome( Helper.*)?$", r"^Safari$", r"^[Oo]llama.*")
PORTS = (8000, 8081)


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def reasons(max_swap_mb: float = 64.0) -> list[str]:
    out = []
    src = power_source()
    if src != "AC":
        out.append(f"not on AC power (power source: {src or 'unknown'})")
    swap = swap_used_mb()
    if swap is None:
        out.append("could not read swap usage")
    elif swap > max_swap_mb:
        out.append(f"swap used {swap:.1f} MB > {max_swap_mb:g} MB (restart the Mac first)")
    pats = [re.compile(p) for p in FORBIDDEN]
    hits = sorted({n for n in running_process_names() for p in pats if p.search(n)})
    if hits:
        out.append(
            "forbidden processes running: "
            + ", ".join(hits[:5])
            + (" ..." if len(hits) > 5 else "")
        )
    for port in PORTS:
        if port_in_use(port):
            out.append(f"port {port} already in use (stop the running engine/gateway first)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Clean-environment preflight")
    ap.add_argument("--max-swap-mb", type=float, default=64.0)
    args = ap.parse_args()
    rs = reasons(args.max_swap_mb)
    for r in rs:
        print(f"preflight FAIL: {r}", file=sys.stderr)
    if rs:
        return 2
    print("preflight ok: AC power, swap within limit, no browsers/Ollama, ports free")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
