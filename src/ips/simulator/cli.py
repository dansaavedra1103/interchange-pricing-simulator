"""Command line: ``python -m ips.simulator run --scenario <file>`` or ``run --all``.

Every scenario file is validated before the base year is read, so a typo fails in a second
rather than after the pricing. Each run prints its one-sentence headline, stops if the fees or
the moved volume no longer balance, and writes its tables under ``data/artifacts/scenarios``.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from ips.simulator.engine import Simulator
from ips.simulator.scenario import load_scenario, scenario_paths
from ips.simulator.state import load_state
from ips.utils.config import load_config
from ips.utils.io import artifacts_dir
from ips.utils.logging import get_logger

SCENARIO_ARTIFACTS = Path("scenarios")


def build_parser() -> argparse.ArgumentParser:
    """The ``run`` command and its options."""
    parser = argparse.ArgumentParser(
        prog="python -m ips.simulator", description="Run interchange pricing scenarios."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run one scenario, or every scenario file.")
    target = run.add_mutually_exclusive_group(required=True)
    target.add_argument("--scenario", type=Path, help="A scenario YAML file.")
    target.add_argument("--all", action="store_true", help="Every file in simulator.scenarios_dir.")
    run.add_argument("--no-write", action="store_true", help="Print without writing artifacts.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the scenarios asked for; returns the exit code."""
    options = build_parser().parse_args(argv)
    logger = get_logger(__name__)
    cfg = load_config()
    paths = scenario_paths(cfg) if options.all else [options.scenario]
    scenarios = [load_scenario(path, cfg) for path in paths]

    start = time.perf_counter()
    simulator = Simulator(load_state(cfg))
    logger.info("base year priced and calibrated in %.1f s", time.perf_counter() - start)
    directory = artifacts_dir(cfg) / SCENARIO_ARTIFACTS
    summaries = []
    for scenario in scenarios:
        result = simulator.run(scenario)
        if not result.conserves():
            raise RuntimeError(
                f"Scenario {scenario.name!r} broke a conservation identity: fee gap "
                f"{result.fee_gap():.2e}, volume gap {result.volume_gap():.2e}"
            )
        print(result.headline())
        if not options.no_write:
            result.write(directory)
        summaries.append(result.summary())

    summary = pl.DataFrame(summaries)
    if not options.no_write:
        directory.mkdir(parents=True, exist_ok=True)
        summary.write_parquet(directory / "summary.parquet")
    with pl.Config(tbl_rows=len(summaries), tbl_width_chars=220, tbl_cols=8):
        print(summary.drop("headline"))
    print(f"{len(scenarios)} scenarios in {time.perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
