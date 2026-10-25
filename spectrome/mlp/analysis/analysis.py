"""Verify saved evaluation results, then write their tables, figures, and report."""

from pathlib import Path

import torch

from .helpers import (
    load_run_config, load_loocv_run, load_nested_run, load_saved_object,
    loocv_report_rows, nested_report_rows, write_report_tables,
    loocv_summary_lines, nested_summary_lines,
)
from .plots import plot_results, plot_nested_results


@torch.no_grad()
def export_results(results, cache, output_dir):
    """Export checked LOOCV subject, region, history, and filter tables."""
    return write_report_tables(output_dir, loocv_report_rows(results, cache))


def _write_nested_report(run_dir, config, results, verified_rows):
    """Produce analysis outputs from verified nested outer-fold evidence."""
    # Export inner selection evidence separately from outer held-out scores.
    tables = write_report_tables(run_dir / "tables", nested_report_rows(results, verified_rows))
    cache = load_saved_object(run_dir / config["cache_path"])
    figures = plot_nested_results(run_dir, results, cache, run_dir / "figures")
    summary = run_dir / "summary.md"
    summary.write_text("\n".join(nested_summary_lines(run_dir, results, verified_rows)), encoding="utf-8")
    return {"tables": tables, "figures": figures, "summary": summary}


def _write_loocv_report(run_dir, config, results, cache):
    """Produce analysis outputs for one verified fixed configuration."""
    tables = export_results(results, cache, run_dir / "tables")
    figures = plot_results(results, cache, run_dir / "figures")
    summary = run_dir / "summary.md"
    summary.write_text("\n".join(loocv_summary_lines(run_dir, config, results)), encoding="utf-8")
    return {"tables": tables, "figures": figures, "summary": summary}


def analyze_run(run_dir):
    """Verify saved predictions before writing any report outputs."""
    run_dir = Path(run_dir)
    config = load_run_config(run_dir)
    if config.get("evaluation") == "nested_cross_validation":
        results, verified_rows = load_nested_run(run_dir, config)
        return _write_nested_report(run_dir, config, results, verified_rows)
    results, cache = load_loocv_run(run_dir, config)
    return _write_loocv_report(run_dir, config, results, cache)
