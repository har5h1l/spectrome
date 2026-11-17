#!/usr/bin/env python3
"""Build a Beamer deck from a finished LOOCV comparison snapshot.

This program formats saved results only. It does not calculate statistics or
read model predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Any


PACKET = Path(__file__).resolve().parents[1]
ROOT = PACKET.parents[1]
ANALYSIS = ROOT / "results" / "analysis" / "loocv-vs-analytical"
DESCRIPTIVE_COLUMNS = (
    "run_id", "hidden_sizes", "activation", "epochs", "learning_rate", "seed",
    "n_subjects", "mean_r", "sd_r", "analytical_mean_r", "analytical_sd_r",
    "mean_gain_r", "sd_gain_r", "n_improved", "run_dir",
)
TEST_COLUMNS = (
    *DESCRIPTIVE_COLUMNS,
    "mean_difference_z", "sd_difference_z", "ci95_low_z", "ci95_high_z",
    "t_statistic", "degrees_of_freedom", "p_raw", "p_bonferroni", "family_size",
)
NUMERIC_COLUMNS = {
    "epochs", "learning_rate", "n_subjects", "mean_r", "sd_r",
    "analytical_mean_r", "analytical_sd_r", "mean_gain_r", "sd_gain_r",
    "n_improved", "mean_difference_z", "sd_difference_z", "ci95_low_z",
    "ci95_high_z", "t_statistic", "degrees_of_freedom", "p_raw",
    "p_bonferroni", "family_size",
}
OPTIONAL_TEST_COLUMNS = {"ci95_low_z", "ci95_high_z", "t_statistic", "p_raw", "p_bonferroni"}


def read_csv(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Required input does not exist: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    return rows


def load_inputs(snapshot_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    snapshot = ANALYSIS / "snapshots" / snapshot_id
    test_rows = read_csv(snapshot / "comparisons.csv", TEST_COLUMNS)
    family_path = snapshot / "family.json"
    if not family_path.is_file():
        raise ValueError(f"Required input does not exist: {family_path}")
    family = json.loads(family_path.read_text(encoding="utf-8"))
    if family.get("snapshot_id") != snapshot_id:
        raise ValueError("family.json snapshot_id does not match the requested snapshot")

    test_by_id = unique_by_id(test_rows, "comparisons.csv")
    expected_ids = family.get("run_ids")
    if not isinstance(expected_ids, list) or not expected_ids:
        raise ValueError("family.json must contain a non-empty run_ids list")
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("family.json contains duplicate run_ids")
    if set(expected_ids) != set(test_by_id):
        raise ValueError("family.json run_ids do not match comparisons.csv")
    if int(family.get("family_size", -1)) != len(expected_ids):
        raise ValueError("family.json family_size does not match run_ids length")

    packet_rows = []
    for run_id in expected_ids:
        merged: dict[str, Any] = dict(test_by_id[run_id])
        if int(merged["family_size"]) != family["family_size"]:
            raise ValueError(f"{run_id}: comparison family size differs from the manifest")
        for key in NUMERIC_COLUMNS:
            value = merged.get(key)
            if value in (None, "") and key in OPTIONAL_TEST_COLUMNS and merged.get("test_status") == "degenerate_zero_variance":
                merged[key] = None
                continue
            if value in (None, ""):
                raise ValueError(f"{run_id}: missing numeric value for {key}")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{run_id}: invalid numeric value for {key}: {value!r}") from exc
            if not math.isfinite(number):
                raise ValueError(f"{run_id}: non-finite numeric value for {key}")
            merged[key] = number
        merged["hidden_sizes"] = parse_hidden_sizes(merged["hidden_sizes"], run_id)
        packet_rows.append(merged)
    return packet_rows, family


def unique_by_id(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        run_id = row.get("run_id", "").strip()
        if not run_id:
            raise ValueError(f"{label} contains an empty run_id")
        if run_id in output:
            raise ValueError(f"{label} contains duplicate run_id {run_id}")
        output[run_id] = row
    return output


def parse_hidden_sizes(value: str, run_id: str) -> list[int]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = [int(piece.strip()) for piece in value.strip("[]() ").split(",") if piece.strip()]
        except ValueError as exc:
            raise ValueError(f"{run_id}: hidden_sizes must be a list of integers") from exc
    if not isinstance(parsed, list) or not parsed or any(not isinstance(x, int) for x in parsed):
        raise ValueError(f"{run_id}: hidden_sizes must be a non-empty list of integers")
    return parsed


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, float):
        if digits == 0:
            return f"{value:.0f}"
        return f"{value:.{digits}g}"
    return str(value)


def tex(value: Any) -> str:
    escapes = {"_": r"\_", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
               "{": r"\{", "}": r"\}", "\\": r"\textbackslash{}",
               "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(escapes.get(char, char) for char in str(value))


def frame(title: str, body: str) -> str:
    return r"\begin{frame}{" + tex(title) + "}\n" + body + "\n" + r"\end{frame}" + "\n"


def table(columns: str, headings: list[str], rows: list[list[str]]) -> str:
    return "\n".join([
        r"\centering\small\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.23}",
        r"\begin{tabular}{" + columns + "}",
        r"\toprule", " & ".join(headings) + r" \\", r"\midrule",
        *(" & ".join(row) + r" \\" for row in rows),
        r"\bottomrule", r"\end{tabular}",
    ])


def run_label(row: dict[str, Any]) -> str:
    return row["run_id"].removeprefix("basic_mlp_")


def architecture(row: dict[str, Any]) -> str:
    return r" $\times$ ".join(str(width) for width in row["hidden_sizes"])


def run_order(row: dict[str, Any]) -> tuple[bool, int, str]:
    label = run_label(row)
    return (False, int(label), "") if label.isdecimal() else (True, 0, label)


def run_slide(row: dict[str, Any]) -> str:
    figures = ROOT / row["run_dir"] / "figures/summary"
    body = [
        rf"\small {architecture(row)} hidden units $\cdot$ {tex(row['activation'])} "
        rf"$\cdot$ {int(row['epochs']):,} epochs $\cdot$ Adam LR {fmt(row['learning_rate'])} "
        rf"$\cdot$ seed {tex(row['seed'])}",
        r"\begin{columns}[T,onlytextwidth]",
    ]
    for label, name in [("Training loss across folds", "training_curves.png"),
                        ("Normalized spectra across subjects", "spectra_across_subjects.png")]:
        image = figures / name
        if not image.is_file():
            raise FileNotFoundError(f"{row['run_id']}: missing {image}")
        body.extend([
            r"\begin{column}{.49\textwidth}\centering",
            r"\small\textbf{" + label + r"}\par",
            r"\includegraphics[width=\linewidth,height=.46\textheight,keepaspectratio]"
            r"{\detokenize{" + str(image) + "}}",
            r"\end{column}",
        ])
    body.extend([
        r"\end{columns}\smallskip",
        rf"\textbf{{Mean held-out $r$}} {row['mean_r']:.5f} (SD {row['sd_r']:.5f})"
        rf"\qquad \textbf{{Gain}} {row['mean_gain_r']:+.5f}"
        rf"\qquad \textbf{{Improved}} {fmt(row['n_improved'], 0)}/{fmt(row['n_subjects'], 0)}",
        r"\par\smallskip",
        rf"\textbf{{Paired Fisher-$z$ test}} $t({fmt(row['degrees_of_freedom'], 0)})={fmt(row['t_statistic'], 4)}$"
        rf"\quad Raw $p$: {fmt(row['p_raw'], 4)}\quad Adjusted $p$: {fmt(row['p_bonferroni'], 4)}",
        r"\par\footnotesize",
        rf"Mean difference {fmt(row['mean_difference_z'], 4)} (SD {fmt(row['sd_difference_z'], 4)})"
        rf"\quad Unadjusted 95\% CI [{fmt(row['ci95_low_z'], 4)}, {fmt(row['ci95_high_z'], 4)}]",
    ])
    return frame(f"Run {run_label(row)}", "\n".join(body))


def make_deck(snapshot_id: str, rows: list[dict[str, Any]], family: dict[str, Any]) -> str:
    by_run = sorted(rows, key=run_order)
    top, lowest = rows[0], rows[-1]
    n = int(top["n_subjects"])
    alpha = float(family["alpha"])
    positive = ", ".join(tex(run_label(row)) for row in by_run
                         if row["mean_difference_z"] > 0 and row["p_bonferroni"] is not None
                         and row["p_bonferroni"] <= alpha) or "None"

    slides = [
        r"\title{Results Packet}" + "\n" +
        r"\subtitle{Frequency-only MLP graph filters}" + "\n" +
        r"\date{Snapshot " + tex(snapshot_id) + "}\n" +
        r"\begin{frame}[plain]\titlepage\end{frame}" + "\n",
        frame("Fixed LOOCV setup", "\n".join([
            r"\begin{itemize}",
            rf"\item \textbf{{{n} subjects.}} Train on {n - 1} subjects and score the held-out subject. Repeat for each subject.",
            rf"\item \textbf{{{family['family_size']} fixed MLP configurations.}} Use frequency as the input and score at the final scheduled epoch.",
            r"\item \textbf{Common baseline.} Score the analytical filter on the same subjects and pair the scores by subject.",
            rf"\item Compare each MLP with the baseline using a two-sided paired Fisher-$z$ test. Adjust across all {family['family_size']} comparisons with Bonferroni.",
            r"\end{itemize}",
        ])),
    ]
    for start in range(0, len(by_run), 8):
        title = "MLP configurations (continued)" if start else "MLP configurations"
        slides.append(frame(title, table(
            "llrrrr", ["Run", "Hidden layers", "Activation", "Epochs", "Learning rate", "Seed"],
            [[tex(run_label(row)), architecture(row), tex(row["activation"]), f"{int(row['epochs']):,}",
              fmt(row["learning_rate"]), tex(row["seed"])] for row in by_run[start:start + 8]],
        )))

    slides.append(frame("Results summary", "\n".join([
        r"\begin{itemize}",
        rf"\item \textbf{{Best observed: Run {tex(run_label(top))}.}} Mean held-out $r={top['mean_r']:.5f}$ "
        rf"versus {top['analytical_mean_r']:.5f} for the analytical filter.",
        rf"\item Run {tex(run_label(top))} gained {top['mean_gain_r']:+.5f} on average and improved "
        rf"{fmt(top['n_improved'], 0)} of {n} subjects. Adjusted $p={fmt(top['p_bonferroni'], 4)}$.",
        rf"\item \textbf{{Lowest observed: Run {tex(run_label(lowest))}.}} Mean held-out $r={lowest['mean_r']:.5f}$ "
        rf"({lowest['mean_gain_r']:+.5f} versus analytical).",
        rf"\item Runs {positive} showed positive paired differences with Bonferroni-adjusted $p\leq {alpha:g}$.",
        r"\end{itemize}",
        r"\small Exploratory comparisons on the development cohort.",
    ])))
    for start in range(0, len(rows), 8):
        title = "Overall comparison (continued)" if start else "Overall comparison"
        slides.append(frame(title, table(
            "rlrrrrr", ["Rank", "Run", r"Mean $r$ (SD)", r"Gain in $r$ (SD)", "Improved", r"Raw $p$", r"Adjusted $p$"],
            [[str(rank), tex(run_label(row)),
              f"{row['mean_r']:.5f} ({row['sd_r']:.5f})",
              f"{row['mean_gain_r']:+.5f} ({row['sd_gain_r']:.5f})",
              f"{fmt(row['n_improved'], 0)}/{fmt(row['n_subjects'], 0)}",
              fmt(row["p_raw"], 4), fmt(row["p_bonferroni"], 4)]
             for rank, row in enumerate(rows[start:start + 8], start + 1)],
        )))
    slides.extend(run_slide(row) for row in by_run)
    slides.append(frame("Conclusions", "\n".join([
        r"\begin{itemize}",
        rf"\item \textbf{{Run {tex(run_label(top))} had the highest observed mean.}} "
        rf"Its gain over the analytical filter was ({top['mean_gain_r']:+.5f} in mean held-out $r$).",
        rf"\item Runs {positive} had positive paired differences after Bonferroni correction "
        rf"across the {family['family_size']} tested configurations.",
        r"\item Repeat the leading frequency-only models with matched seeds to check sensitivity to initialization.",
        r"\item Evaluate subject-specific inputs in a later comparison against a matched frequency-only model.",
        r"\end{itemize}",
        r"\small These results describe the current cohort and training settings.",
    ])))
    return PREAMBLE + "\n".join(slides) + "\n" + r"\end{document}" + "\n"


PREAMBLE = r"""\documentclass[aspectratio=169,11pt]{beamer}
\usepackage{booktabs}
\definecolor{ink}{HTML}{17343C}
\definecolor{accent}{HTML}{1A6B65}
\setbeamercolor{normal text}{fg=ink,bg=white}
\setbeamercolor{structure}{fg=accent}
\setbeamercolor{frametitle}{fg=accent}
\setbeamerfont{frametitle}{series=\bfseries,size=\Large}
\setbeamerfont{title}{series=\bfseries,size=\Huge}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{footline}{\hfill\small\insertframenumber\,/\,\inserttotalframenumber\hspace{1em}\vspace{.6em}}
\begin{document}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_id")
    parser.add_argument("--output", type=Path, help="Destination PDF")
    args = parser.parse_args()
    if not args.snapshot_id or Path(args.snapshot_id).name != args.snapshot_id or args.snapshot_id in {".", ".."}:
        parser.error("snapshot_id must be a snapshot directory name")
    output = (args.output or PACKET / "presentation.pdf").resolve()
    if output.suffix != ".pdf":
        parser.error("--output must end in .pdf")
    try:
        rows, family = load_inputs(args.snapshot_id)
        source = make_deck(args.snapshot_id, rows, family)
        with TemporaryDirectory(prefix="spectrome-beamer-") as temporary:
            work = Path(temporary)
            (work / "packet.tex").write_text(source, encoding="utf-8")
            subprocess.run(["latexmk", "-pdf", "-halt-on-error", "-interaction=nonstopmode", "packet.tex"],
                           check=True, cwd=work)
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(work / "packet.pdf", output)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
