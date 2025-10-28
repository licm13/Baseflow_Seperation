"""Utility entry point for executing multiple batch separation pipelines.

该脚本旨在为调度系统或人工值守场景提供统一入口，可一次执行多种
时间尺度的批处理流程。通过命令行参数即可灵活选择运行的脚本、循环
间隔与是否捕获错误。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List

# 默认批处理脚本按时间尺度划分
DEFAULT_SCRIPTS = [
    Path("batch/daily_batch_run.py"),
    Path("batch/monthly_batch_run.py"),
    Path("batch/long_term_batch_run.py"),
]


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute one or more baseflow batch-processing scripts sequentially. "
            "Relative paths are resolved against the scripts package directory."
        )
    )
    parser.add_argument(
        "scripts",
        nargs="*",
        type=Path,
        default=DEFAULT_SCRIPTS,
        help=(
            "脚本列表，默认依次执行逐日、逐月与长时序批处理脚本。"
            "既可以传入文件名，也可以传入相对 scripts/ 的路径。"
        ),
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=0,
        help="循环执行时的间隔秒数；默认为 0 表示只运行一轮。",
    )
    parser.add_argument(
        "--halt-on-error",
        action="store_true",
        help="遇到子进程异常时立即终止（默认会持续循环并记录错误）。",
    )
    return parser.parse_args(argv)


def _resolve_scripts(script_paths: Iterable[Path]) -> List[Path]:
    base_dir = Path(__file__).resolve().parent
    resolved = []
    for script in script_paths:
        candidate = script if script.is_absolute() else base_dir / script
        if not candidate.exists():
            raise FileNotFoundError(f"无法找到脚本: {candidate}")
        resolved.append(candidate)
    return resolved


def run_scripts(script_paths: Iterable[Path]) -> None:
    """Sequentially execute the provided Python scripts."""

    python_executable = Path(sys.executable)
    for script in script_paths:
        print(f"\n=== Running {script.name} ===")
        subprocess.run([str(python_executable), str(script)], check=True)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    scripts = _resolve_scripts(args.scripts)

    while True:
        try:
            run_scripts(scripts)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - 调度异常
            print(f"[ERROR] {exc}. 使用 --halt-on-error 可选择终止循环。")
            if args.halt_on_error:
                raise
        if args.interval <= 0:
            break
        time.sleep(args.interval)


if __name__ == "__main__":  # pragma: no cover - 脚本入口
    main()
