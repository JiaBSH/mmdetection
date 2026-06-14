"""Unified CLI entry point for mmdet_utils.

Usage:
    python -m mmdet_utils get-weight <config_path>
    python -m mmdet_utils check-completed <work_dir>
    python -m mmdet_utils extract-failure <log_file>
    python -m mmdet_utils check-weight-load <log_file> <weights_source>
    python -m mmdet_utils collect-stats <config_path> <work_dir>
    python -m mmdet_utils display <summary_tsv> --stats <stats_tsv>

All commands write results to stdout and errors/warnings to stderr.
Exit code 0 on success, 1 on error.
"""

from __future__ import annotations

import argparse
import sys


def _cmd_get_weight(args: argparse.Namespace) -> None:
    from .config import get_weight_info

    source, detail = get_weight_info(args.config_path)
    print(f"{source}\t{detail}")


def _cmd_check_completed(args: argparse.Namespace) -> None:
    from .completed import is_model_completed

    if is_model_completed(args.work_dir):
        sys.exit(0)
    else:
        sys.exit(1)


def _cmd_extract_failure(args: argparse.Namespace) -> None:
    from .log_parser import extract_failure_reason

    reason = extract_failure_reason(args.log_file)
    print(reason)


def _cmd_check_weight_load(args: argparse.Namespace) -> None:
    from .log_parser import detect_weight_load_status

    status = detect_weight_load_status(args.log_file, args.weights_source)
    print(status)


def _cmd_collect_stats(args: argparse.Namespace) -> None:
    from .stats import collect_model_stats, format_stats_line

    stats = collect_model_stats(args.config_path, args.work_dir)
    print(format_stats_line(stats))


def _cmd_display(args: argparse.Namespace) -> None:
    from .display import print_all

    print_all(args.summary_tsv, args.stats)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m mmdet_utils",
        description="mmdetection batch training utilities",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # get-weight
    p = subparsers.add_parser("get-weight", help="Extract weight info from config")
    p.add_argument("config_path", help="Path to mmdet config .py file")
    p.set_defaults(func=_cmd_get_weight)

    # check-completed
    p = subparsers.add_parser("check-completed", help="Check if model already trained")
    p.add_argument("work_dir", help="Training work directory")
    p.set_defaults(func=_cmd_check_completed)

    # extract-failure
    p = subparsers.add_parser("extract-failure", help="Extract failure reason from log")
    p.add_argument("log_file", help="Path to run.log")
    p.set_defaults(func=_cmd_extract_failure)

    # check-weight-load
    p = subparsers.add_parser("check-weight-load", help="Detect weight load status")
    p.add_argument("log_file", help="Path to run.log")
    p.add_argument("weights_source", help="Weight source: load_from, init_cfg, none, etc.")
    p.set_defaults(func=_cmd_check_weight_load)

    # collect-stats
    p = subparsers.add_parser("collect-stats", help="Collect model statistics")
    p.add_argument("config_path", help="Path to mmdet config .py file")
    p.add_argument("work_dir", help="Training work directory")
    p.set_defaults(func=_cmd_collect_stats)

    # display
    p = subparsers.add_parser("display", help="Print formatted summary and stats tables")
    p.add_argument("summary_tsv", help="Path to run_summary.tsv")
    p.add_argument("--stats", default=None, help="Path to model_stats.tsv")
    p.set_defaults(func=_cmd_display)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
