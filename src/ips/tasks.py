"""Project tasks, runnable on any operating system: ``python -m ips.tasks <task>``.

    generate [--sample]   synthetic data into data/raw (Feature 1)
    dbt [args ...]        dbt build on DuckDB, or any dbt command (dbt run --select marts)
    docs [--serve]        dbt docs generate, and optionally serve the lineage in the browser
    segment [--months N]  merchant segmentation on the graph, against the baseline (Feature 4)
    test                  pytest -q, ruff check and ruff format --check
    all [--sample]        generate, dbt build and test

The Makefile delegates here, so Linux and CI can keep using ``make``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence

from ips.utils.config import load_config, project_root
from ips.utils.io import raw_dir, run_dbt

_CHECKS = (["pytest", "-q"], ["ruff", "check", "."], ["ruff", "format", "--check", "."])


def _generate(sample: bool) -> int:
    from ips.data_gen.generate import main as generate_main

    generate_main(["--sample"] if sample else [])
    return 0


def _dbt(args: Sequence[str]) -> int:
    # Las tablas de referencia (tarifas, fees, MCC) salen del config: se reescriben antes de
    # cada corrida, así cambiar un fee no exige regenerar las transacciones.
    from ips.economics.params import write_reference_tables

    write_reference_tables(load_config(), raw_dir())
    return run_dbt(list(args) or ["build"]).returncode


def _docs(serve: bool) -> int:
    code = run_dbt(["docs", "generate"]).returncode
    if code == 0 and serve:
        code = run_dbt(["docs", "serve"]).returncode
    return code


def _segment(months: int | None) -> int:
    from ips.graph.pipeline import run_segmentation, write_artifacts

    cfg = load_config()
    run = run_segmentation(cfg, months=months)
    paths = write_artifacts(run, cfg)
    print(run.comparison)
    print(run.marginal)
    for sentence in run.verdicts():
        print(sentence)
    print("artifacts: " + ", ".join(str(path) for path in paths.values()))
    return 0


def _test() -> int:
    root = str(project_root())
    for command in _CHECKS:
        code = subprocess.run([sys.executable, "-m", *command], cwd=root, check=False).returncode
        if code != 0:
            return code
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the task and run it; returns the exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m ips.tasks", description="Interchange Pricing Simulator tasks."
    )
    tasks = parser.add_subparsers(dest="task", required=True)
    generate = tasks.add_parser("generate", help="Generate the synthetic dataset.")
    generate.add_argument("--sample", action="store_true", help="Use sample_sizes (tests, CI).")
    dbt = tasks.add_parser("dbt", help="Run dbt against DuckDB (default: dbt build).")
    dbt.add_argument("args", nargs=argparse.REMAINDER, help="dbt command and flags.")
    docs = tasks.add_parser("docs", help="Generate the dbt docs (lineage).")
    docs.add_argument("--serve", action="store_true", help="Serve them in the browser.")
    segment = tasks.add_parser("segment", help="Segment merchants on the graph (Feature 4).")
    segment.add_argument("--months", type=int, default=None, help="Window, in months.")
    tasks.add_parser("test", help="Run pytest and ruff.")
    run_all = tasks.add_parser("all", help="Generate, build and test.")
    run_all.add_argument("--sample", action="store_true", help="Use sample_sizes.")
    options = parser.parse_args(argv)

    if options.task == "generate":
        return _generate(options.sample)
    if options.task == "dbt":
        return _dbt(options.args)
    if options.task == "docs":
        return _docs(options.serve)
    if options.task == "segment":
        return _segment(options.months)
    if options.task == "test":
        return _test()
    for step in (lambda: _generate(options.sample), lambda: _dbt([]), _test):
        code = step()
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
