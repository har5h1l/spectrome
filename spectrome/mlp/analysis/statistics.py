"""Paired Fisher r-to-z statistics for saved subject-level results."""

import math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from scipy import stats

from ..spectra import SCORING_ID
from ..training.helpers import save_json_atomic
from .helpers import load_loocv_scores


def _score_map(rows, label):
    if isinstance(rows, dict):
        values = dict(rows)
        if len(values) != len(rows):
            raise ValueError(f"{label}: duplicate subject IDs")
        return values
    values = {}
    for row in rows:
        subject_id = row.get("subject_id")
        if not isinstance(subject_id, str) or not subject_id:
            raise ValueError(f"{label}: every score row needs a nonempty subject_id")
        if subject_id in values:
            raise ValueError(f"{label}: duplicate score for {subject_id}")
        values[subject_id] = row.get("r")
    return values


def paired_fisher_z_test(left_scores, right_scores, *, left_label, right_label,
                         contrast, expected_subject_ids=None,
                         provenance=None, adjustment_count=None):
    """Compare paired subject mean-r scores after Fisher's arctanh transform.

    The values are already mean regional Pearson correlations, so each subject
    contributes one paired observation. Endpoints and nonfinite values are
    rejected rather than clipped.
    """
    if (not isinstance(left_label, str) or not left_label
            or not isinstance(right_label, str) or not right_label
            or left_label == right_label):
        raise ValueError("Paired score labels must be distinct nonempty strings")
    left = _score_map(left_scores, left_label)
    right = _score_map(right_scores, right_label)
    if expected_subject_ids is None:
        expected = sorted(set(left) | set(right))
    else:
        expected = list(expected_subject_ids)
        if len(expected) != len(set(expected)):
            raise ValueError("expected_subject_ids must be unique")
    if set(left) != set(expected) or set(right) != set(expected):
        missing_left = sorted(set(expected) - set(left))
        missing_right = sorted(set(expected) - set(right))
        extra_left = sorted(set(left) - set(expected))
        extra_right = sorted(set(right) - set(expected))
        raise ValueError(
            "Paired scores must contain exactly the expected subject IDs "
            f"(missing left={missing_left}, missing right={missing_right}, "
            f"extra left={extra_left}, extra right={extra_right})"
        )
    if len(expected) < 2:
        raise ValueError("A paired t-test requires at least two subjects")

    left_r, right_r = [], []
    for subject_id in expected:
        for label, value in ((left_label, left[subject_id]), (right_label, right[subject_id])):
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{label}: correlation for {subject_id} is not numeric") from error
            if not math.isfinite(number) or not -1 < number < 1:
                raise ValueError(
                    f"{label}: correlation for {subject_id} must be finite and strictly between -1 and 1"
                )
            (left_r if label == left_label else right_r).append(number)

    left_r = np.asarray(left_r, dtype=np.float64)
    right_r = np.asarray(right_r, dtype=np.float64)
    left_z = np.arctanh(left_r)
    right_z = np.arctanh(right_r)
    z_difference = left_z - right_z
    raw_difference = left_r - right_r
    n = len(expected)
    df = n - 1
    mean_z = float(z_difference.mean())
    sd_z = float(z_difference.std(ddof=1))
    standard_error_z = float(sd_z / math.sqrt(n))

    if np.all(z_difference == z_difference[0]):
        statistic = pvalue = ci_low = ci_high = None
        status = "degenerate_zero_variance"
        diagnostic = (
            "All paired Fisher-z differences are identical, so the sample "
            "standard error is zero and the t statistic is undefined."
        )
    else:
        test = stats.ttest_rel(left_z, right_z, alternative="two-sided")
        statistic = float(test.statistic)
        pvalue = float(test.pvalue)
        critical = float(stats.t.ppf(0.975, df))
        ci_low = mean_z - critical * standard_error_z
        ci_high = mean_z + critical * standard_error_z
        status = "ok"
        diagnostic = None

    adjusted_pvalue = None
    if pvalue is not None and adjustment_count is not None:
        if isinstance(adjustment_count, bool) or not isinstance(adjustment_count, int) or adjustment_count < 1:
            raise ValueError("adjustment_count must be a positive integer or None")
        adjusted_pvalue = min(1.0, pvalue * adjustment_count)

    subject_rows = [
        {
            "subject_id": subject_id,
            f"{left_label}_r": float(left_r[index]),
            f"{right_label}_r": float(right_r[index]),
            f"{left_label}_fisher_z": float(left_z[index]),
            f"{right_label}_fisher_z": float(right_z[index]),
            "fisher_z_difference": float(z_difference[index]),
            "raw_r_difference": float(raw_difference[index]),
        }
        for index, subject_id in enumerate(expected)
    ]
    return {
        "contrast": contrast,
        "left_label": left_label,
        "right_label": right_label,
        "method": "two-sided paired t-test on per-subject Fisher z of mean regional r",
        "status": status,
        "diagnostic": diagnostic,
        "n_subject_pairs": n,
        "degrees_of_freedom": df,
        "t_statistic": statistic,
        "p_value_unadjusted": pvalue,
        "adjustment": (None if adjustment_count is None else {
            "method": "Bonferroni",
            "planned_tests": adjustment_count,
            "p_value": adjusted_pvalue,
        }),
        "raw_r_gain_mean": float(raw_difference.mean()),
        "raw_r_gain_sd": float(raw_difference.std(ddof=1)),
        "fisher_z_difference_mean": mean_z,
        "fisher_z_difference_sd": sd_z,
        "fisher_z_difference_standard_error": standard_error_z,
        "fisher_z_difference_95_percent_ci": (
            None if ci_low is None else {"scale": "Fisher z difference", "low": ci_low, "high": ci_high}
        ),
        "provenance": provenance or {},
        "independence_note": (
            "Subjects are the paired sampling unit. Cross-validation training sets overlap; "
            "these p-values are exploratory and do not remove model-selection bias."
        ),
        "subject_rows": subject_rows,
    }




def _require_matching_provenance(left, right):
    keys = ("cache_sha256", "preprocessing_id", "forward_sha256", "frequencies_hz",
            "subject_ids", "observed_region_maps", "scoring_id")
    mismatches = [key for key in keys if left["provenance"].get(key) != right["provenance"].get(key)]
    if mismatches:
        raise ValueError(f"Runs have incompatible cohort/scoring provenance: {', '.join(mismatches)}")


def compare_saved_runs(left_run, right_run, *, contrast, left_label=None, right_label=None,
                       expected_subject_ids=None, adjustment_count=None):
    """Compare the learned scores from two saved fixed-configuration run folders."""
    left = load_loocv_scores(left_run)
    right = load_loocv_scores(right_run)
    _require_matching_provenance(left, right)
    expected = expected_subject_ids or left["provenance"]["subject_ids"]
    return paired_fisher_z_test(
        left["learned"], right["learned"],
        left_label=left_label or Path(left_run).name,
        right_label=right_label or Path(right_run).name,
        contrast=contrast,
        expected_subject_ids=expected,
        provenance={"left": left["provenance"], "right": right["provenance"]},
        adjustment_count=adjustment_count,
    )


def compare_saved_run_to_analytical(run_dir, *, contrast, model_label=None,
                                    expected_subject_ids=None, adjustment_count=None):
    """Compare one saved run's MLP and analytical scores for matched subjects."""
    run = load_loocv_scores(run_dir)
    expected = expected_subject_ids or run["provenance"]["subject_ids"]
    return paired_fisher_z_test(
        run["learned"], run["analytical"],
        left_label=model_label or Path(run_dir).name,
        right_label="analytical",
        contrast=contrast,
        expected_subject_ids=expected,
        provenance={"run": run["provenance"]},
        adjustment_count=adjustment_count,
    )


def save_retrospective_comparisons(output_path, comparisons):
    """Save named comparison results and all paired subject values as JSON."""
    names = [item.get("contrast") for item in comparisons]
    if not comparisons or any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("comparisons must have unique named contrast records")
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "retrospective_exploratory_subject_level_comparison",
        "method": "Fisher arctanh of each subject's mean regional r; two-sided paired t-test",
        "multiple_testing_adjustment": None,
        "comparisons": comparisons,
    }
    save_json_atomic(output_path, payload)
    return Path(output_path)
