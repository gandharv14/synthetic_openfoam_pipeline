from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run_command(
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 120,
    env: dict[str, str] | None = None,
) -> CommandResult:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.run(
        args,
        cwd=cwd,
        env=merged_env,
        timeout=timeout,
        text=True,
        capture_output=True,
        check=False,
    )
    return CommandResult(args=args, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)

