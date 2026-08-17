'''
Tests for correlate_data.py. See CORRELATION_APP_SPEC.md section 8 for the source of
each test's expected values (row counts verified directly against the real
combined_results_with_anchor.csv, not invented).

Tests that need the real anchor-merged CSV look for it at the path in
BOLTZ_CORRELATE_TEST_CSV (env var), falling back to
"combined_results_with_anchor.csv" next to this file, and skip (not fail) if neither
exists -- that file is real project data, not a fixture checked into this repo.
'''

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import correlate_data as cd

_CSV_PATH = os.environ.get(
    "BOLTZ_CORRELATE_TEST_CSV",
    str(Path(__file__).parent / "combined_results_with_anchor.csv"),
)
_HAS_REAL_CSV = Path(_CSV_PATH).exists()

requires_real_csv = pytest.mark.skipif(
    not _HAS_REAL_CSV,
    reason=(
        f"Real anchor-merged CSV not found at '{_CSV_PATH}' -- set BOLTZ_CORRELATE_TEST_CSV "
        f"to its path to run this test"
    ),
)


@pytest.fixture(scope="module")
def real_df():
    return cd.load_data(_CSV_PATH)


# --- Test 1: find_receptors -------------------------------------------------------

@requires_real_csv
def test_find_receptors_on_real_csv(real_df):
    assert cd.find_receptors(real_df) == ["M1", "M2", "M3", "M4", "M5"]


def test_receptor_sort_key_handles_non_numeric_without_raising():
    receptors = ["other", "M10", "M2", "M1", "custom"]
    assert sorted(receptors, key=cd._receptor_sort_key) == ["M1", "M2", "M10", "custom", "other"]


# --- Test 2: target matching on named compounds -----------------------------------

@requires_real_csv
def test_sigma1_compound_excluded_from_every_receptor_ki_series(real_df):
    row = real_df[real_df["chembl_id"] == "CHEMBL445102"]
    assert len(row) == 1
    for receptor in ["M1", "M2", "M3", "M4", "M5"]:
        mask = cd.match_target(real_df, receptor, "Ki")
        assert mask.loc[row.index[0]] == False, f"CHEMBL445102 should not match {receptor} Ki"


@requires_real_csv
def test_m2_compound_matches_only_m2_ki_series(real_df):
    row = real_df[real_df["chembl_id"] == "CHEMBL422763"]
    assert len(row) == 1
    assert row.iloc[0]["Ki_uniprot"] == "P08172"
    assert row.iloc[0]["best_ki_nm_y"] == pytest.approx(2.692)
    for receptor in ["M1", "M2", "M3", "M4", "M5"]:
        mask = cd.match_target(real_df, receptor, "Ki")
        assert mask.loc[row.index[0]] == (receptor == "M2")


@requires_real_csv
def test_adenosine_a1_compound_excluded_from_m5(real_df):
    row = real_df[real_df["chembl_id"] == "CHEMBL4572905"]
    assert len(row) == 1
    assert row.iloc[0]["Ki_uniprot"] == "P30542"
    mask = cd.match_target(real_df, "M5", "Ki")
    assert mask.loc[row.index[0]] == False


# --- Test 2b: row counts after target matching equal the spec's table exactly ----

_EXPECTED_ROW_COUNTS = {
    "M1": {"Ki": 273, "IC50": 292, "EC50": 263},
    "M2": {"Ki": 154, "IC50": 54, "EC50": 53},
    "M3": {"Ki": 189, "IC50": 89, "EC50": 9},
    "M4": {"Ki": 94, "IC50": 33, "EC50": 203},
    "M5": {"Ki": 31, "IC50": 48, "EC50": 29},
}


@requires_real_csv
@pytest.mark.parametrize("receptor", ["M1", "M2", "M3", "M4", "M5"])
def test_row_counts_match_spec_table(real_df, receptor):
    for activity, (value_col, _, _) in cd.ACTIVITY_COLUMNS.items():
        mask = cd.match_target(real_df, receptor, activity) & real_df[value_col].notna()
        expected = _EXPECTED_ROW_COUNTS[receptor][activity]
        assert int(mask.sum()) == expected, f"{receptor}/{activity}: got {int(mask.sum())}, expected {expected}"


# --- Test 2c: no best_*_nm_x column is ever read ----------------------------------

def test_activity_columns_never_reference_x_columns():
    x_columns = [value_col for value_col, _, _ in cd.ACTIVITY_COLUMNS.values() if value_col.endswith('_x')]
    assert x_columns == []


@requires_real_csv
def test_matching_unchanged_when_x_columns_are_dropped(real_df):
    df_no_x = real_df.drop(columns=[c for c in real_df.columns if c.startswith('best_') and c.endswith('_nm_x')])
    for receptor in cd.find_receptors(real_df):
        for activity, (value_col, _, _) in cd.ACTIVITY_COLUMNS.items():
            with_x = cd.match_target(real_df, receptor, activity) & real_df[value_col].notna()
            without_x = cd.match_target(df_no_x, receptor, activity) & df_no_x[value_col].notna()
            assert (with_x.values == without_x.values).all(), f"{receptor}/{activity} differs after dropping _x columns"


# --- Test 4: p-scale conversion ---------------------------------------------------

def test_nm_to_p_reference_points():
    assert cd.nm_to_p(1) == pytest.approx(9.0)
    assert cd.nm_to_p(1000) == pytest.approx(6.0)


def test_pred_to_p_reference_points():
    assert cd.pred_to_p(0.0) == pytest.approx(6.0)
    assert cd.pred_to_p(-1.0) == pytest.approx(7.0)


def test_nm_to_p_rejects_zero_and_negative_without_raising():
    assert np.isnan(cd.nm_to_p(0))
    assert np.isnan(cd.nm_to_p(-5))
    assert not np.isinf(cd.nm_to_p(0))
    assert not np.isinf(cd.nm_to_p(-5))


def test_nm_to_p_works_on_a_pandas_series():
    result = cd.nm_to_p(pd.Series([1, 1000, 0, -5, np.nan]))
    assert result.iloc[0] == pytest.approx(9.0)
    assert result.iloc[1] == pytest.approx(6.0)
    assert np.isnan(result.iloc[2])
    assert np.isnan(result.iloc[3])
    assert np.isnan(result.iloc[4])


@requires_real_csv
def test_chembl422763_m2_pair_p_scale_values(real_df):
    row = real_df[real_df["chembl_id"] == "CHEMBL422763"].iloc[0]
    pred_p = cd.pred_to_p(row["M2_affinity_pred_value"])
    exp_p = cd.nm_to_p(row["best_ki_nm_y"])
    assert round(pred_p, 2) == pytest.approx(6.93)
    assert round(exp_p, 2) == pytest.approx(8.57)


# --- Test 3: per-receptor filtering ------------------------------------------------

def _synthetic_distance_test_df():
    """Minimal synthetic frame using CHEMBL4246159's real M2/M3 distance values
    (13.941 A, 8.403 A -- see the module docstring's step-1 verification). The real
    compound only has EC50 data matched to M1 in the actual dataset (its Ki/IC50 are
    both null), so it never reaches M2's or M3's real paired series regardless of any
    distance filter -- this synthetic frame isolates the filter mechanic itself with
    the same realistic numbers, matched cleanly to M2 and M3 respectively via Ki."""
    return pd.DataFrame({
        "chembl_id": ["TEST_M2_COMPOUND", "TEST_M3_COMPOUND"],
        "Ki_uniprot": ["P08172", "P20309"],
        "Ki_target_name": ["Muscarinic acetylcholine receptor M2", "Muscarinic acetylcholine receptor M3"],
        "best_ki_nm_y": [50.0, 50.0],
        "M2_affinity_pred_value": [-1.0, np.nan],
        "M2_distance_to_orthosteric_site": [13.94095516204834, np.nan],
        "M3_affinity_pred_value": [np.nan, -0.895235538482666],
        "M3_distance_to_orthosteric_site": [np.nan, 8.40293025970459],
    })


def test_distance_filter_drops_far_pose_keeps_close_pose():
    df = _synthetic_distance_test_df()
    config = cd.FilterConfig(max_distance=9.0)

    m2_filtered, m2_report = cd.apply_filters(df, "M2", "Ki", config)
    assert "TEST_M2_COMPOUND" not in m2_filtered["chembl_id"].values
    assert m2_report.final_n == 0
    assert m2_report.initial_n == 1

    m3_filtered, m3_report = cd.apply_filters(df, "M3", "Ki", config)
    assert "TEST_M3_COMPOUND" in m3_filtered["chembl_id"].values
    assert m3_report.final_n == 1


def test_filters_are_independent_per_receptor():
    """A compound excluded for one receptor (bad pose) must still appear for another
    receptor (good pose) -- filtering never drops a row globally."""
    df = pd.DataFrame({
        "chembl_id": ["COMPOUND_X"],
        "Ki_uniprot": ["P08172"],
        "Ki_target_name": ["Muscarinic acetylcholine receptor M2"],
        "best_ki_nm_y": [10.0],
        "IC50_uniprot": ["P08912"],
        "IC50_target_name": ["Muscarinic acetylcholine receptor M5"],
        "best_ic50_nm_y": [10.0],
        "M2_affinity_pred_value": [-1.0],
        "M2_distance_to_orthosteric_site": [5.0],
        "M5_affinity_pred_value": [-1.0],
        "M5_distance_to_orthosteric_site": [20.0],
    })
    config = cd.FilterConfig(max_distance=9.0)

    m2_filtered, _ = cd.apply_filters(df, "M2", "Ki", config)
    assert "COMPOUND_X" in m2_filtered["chembl_id"].values

    m5_filtered, _ = cd.apply_filters(df, "M5", "IC50", config)
    assert "COMPOUND_X" not in m5_filtered["chembl_id"].values


def test_filter_report_tracks_steps_and_final_n():
    df = _synthetic_distance_test_df()
    _, report = cd.apply_filters(df, "M2", "Ki", cd.FilterConfig(max_distance=9.0))
    assert report.receptor == "M2"
    assert report.activity == "Ki"
    assert report.initial_n == 1
    assert report.final_n == 0
    assert len(report.steps) == 1
    label, removed, remaining = report.steps[0]
    assert removed == 1
    assert remaining == 0
    assert "M2 / Ki" in report.as_text()


# --- Test 5/6: regression ----------------------------------------------------------

def test_regression_on_perfect_fit_returns_slope_1_r2_1():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = x.copy()
    result = cd.fit_regression(x, y)
    assert result is not None
    assert result.slope == pytest.approx(1.0)
    assert result.intercept == pytest.approx(0.0, abs=1e-9)
    assert result.r_squared == pytest.approx(1.0)


def test_fit_regression_handles_too_few_points_without_raising():
    assert cd.fit_regression([], []) is None
    assert cd.fit_regression([1.0], [2.0]) is None


def test_fit_regression_handles_zero_variance_without_raising():
    assert cd.fit_regression([1.0, 1.0, 1.0], [2.0, 3.0, 4.0]) is None
    assert cd.fit_regression([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]) is None


# --- Test 4b: p-scale labeling never leaks "nM" into figure text or CSV headers ---

@requires_real_csv
def test_no_nm_string_in_figure_text_or_exported_headers(real_df, tmp_path):
    config = cd.FilterConfig()
    fig, _, _ = cd.make_figure(real_df, ["M2"], "Ki", config, layout='overlay', min_points=1)

    texts = []
    for ax in fig.axes:
        texts.append(ax.get_xlabel())
        texts.append(ax.get_ylabel())
        texts.append(ax.get_title())
        texts.extend(t.get_text() for t in ax.texts)
        legend = ax.get_legend()
        if legend:
            texts.extend(t.get_text() for t in legend.get_texts())
    # Note: the figure's suptitle is deliberately NOT checked here -- it's allowed to
    # echo a user-entered nM filter threshold (e.g. "potency>=10nM") verbatim, since
    # that's the unit the user typed it in. The spec's "never nM" rule is scoped to
    # axis labels, legend entries, and exported column headers (checked above/below).
    for text in texts:
        assert "nM" not in text, f"found 'nM' in figure text: {text!r}"

    csv_path = tmp_path / "export.csv"
    cd.export_paired_csv(real_df, "M2", "Ki", config, str(csv_path))
    header = csv_path.read_text().splitlines()[0]
    assert "nM" not in header


# --- make_figure / export smoke tests ----------------------------------------------

@requires_real_csv
def test_make_figure_overlay_and_facet_layouts(real_df):
    config = cd.FilterConfig()
    fig_overlay, reports_overlay, points_overlay = cd.make_figure(real_df, ["M1", "M2"], "Ki", config, layout='overlay', min_points=1)
    assert len(fig_overlay.axes) == 1
    assert set(reports_overlay) == {"M1", "M2"}
    assert set(points_overlay) == {"M1", "M2"}

    fig_facet, reports_facet, points_facet = cd.make_figure(real_df, ["M1", "M2"], "Ki", config, layout='facet', min_points=1)
    assert len(fig_facet.axes) == 2
    assert set(reports_facet) == {"M1", "M2"}
    assert set(points_facet) == {"M1", "M2"}


@requires_real_csv
def test_export_paired_csv_round_trips(real_df, tmp_path):
    config = cd.FilterConfig()
    csv_path = tmp_path / "pairs.csv"
    export_df = cd.export_paired_csv(real_df, "M2", "Ki", config, str(csv_path))
    assert len(export_df) > 0
    reloaded = pd.read_csv(csv_path)
    assert len(reloaded) == len(export_df)
    assert set(["chembl_id", "receptor", "activity", "predicted_p", "experimental_p"]).issubset(reloaded.columns)
