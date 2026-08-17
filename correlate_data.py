'''
Correlates Boltz-2 predicted affinity against experimental ChEMBL potency (Ki / IC50 /
EC50), per receptor, with mandatory UniProt-based target matching, per-receptor
filtering, p-scale conversion, and OLS/Spearman regression with figure export.

Pure logic + CLI only -- no PyQt import, no plt.show(). Every function takes explicit
arguments and returns values (make_figure() returns a Figure object, not a file path)
so correlate_data_gui.py can call apply_filters()/fit_regression()/make_figure()
directly and embed the result live, rather than round-tripping through a temp-dir PNG.

Expects the output of `collate_combine_boltz2.py --anchor-csv ... --receptor-config ...`
as input (e.g. "combined_results_with_anchor.csv") -- a table joining, on chembl_id:
  - a left block of ChEMBL experimental data (target_name/target_uniprot plus
    best_{ic50,ec50,ki}_nm_x/_y and each activity type's own target_name/target_chembl_id/
    uniprot/assay_count -- see the note on _x vs _y below), and
  - a right block of four Boltz-2 prediction columns per receptor:
    <receptor>_affinity_pred_value, <receptor>_affinity_probability_binary,
    <receptor>_confidence_score, <receptor>_distance_to_orthosteric_site.
Receptors are discovered dynamically from the right block's column names (any name
works, not just "M<number>") -- see find_receptors().

_x vs _y: best_*_nm_x and best_*_nm_y are duplicate potency columns produced by a
name collision in the upstream ChEMBL merge. This pipeline only ever reads the _y
columns -- the _x columns are an upstream artefact that should eventually be dropped
in collate_combine_boltz2.py; load_data() logs a warning if it sees them.

Target matching is mandatory, not optional: most compounds in this dataset were
assayed against non-muscarinic targets (Adenosine A1/A2a, D2, PDE4, Sigma-1, ...), so
a row only contributes to a receptor's series for a given activity type if that
activity's `<activity>_uniprot` (or, when that's null, its `<activity>_target_name`)
identifies the receptor -- see match_target() and RECEPTOR_UNIPROTS. Matching is
independent per activity type: a compound can be Ki-matched to one subtype and
IC50-matched to a different one.

Both axes are always on p-scale (pKi/pIC50/pEC50-equivalent) -- there is no raw-nM
mode. See pred_to_p()/nm_to_p() for the two conversions and why plain nM isn't used.
No axis label, legend entry, or exported CSV column header ever says "nM" -- a raw nM
value is exported for traceability (see export_paired_csv()) under a name that avoids
that exact substring, since it's the one thing most likely to be misread as the axis
unit if it slipped into a label.

Filtering (apply_filters/FilterConfig) is always per-receptor: a compound dropped for
one receptor's bad pose can still appear in another receptor's plot.

Usage:
python correlate_data.py --input combined_results_with_anchor.csv --receptors M1,M2 \\
    --activity Ki --layout overlay --output m1_m2_ki.png
'''

import argparse
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.figure import Figure
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Receptor discovery ---------------------------------------------------------

# Matches any receptor prefix, not just "M<number>" -- receptor names come from
# collate_combine_boltz2.py's --receptor-config keys, which can be arbitrary.
RECEPTOR_AFFINITY_PATTERN = re.compile(r'^(.+)_affinity_pred_value$')


def _receptor_sort_key(receptor: str) -> tuple:
    """Sort key for receptor prefixes: those containing a number (M1, M2, M10, ...) sort
    numerically by it; prefixes with no digit at all sort after all numeric ones,
    alphabetically among themselves -- unlike a bare int(re.search(r'\\d+', r).group()),
    this never raises on a non-numeric receptor name."""
    match = re.search(r'\d+', receptor)
    if match:
        return (0, int(match.group()), receptor)
    return (1, 0, receptor)


def find_receptors(df: pd.DataFrame) -> list:
    """Find receptor prefixes (e.g. 'M1', or any other name used in --receptor-config)
    that have an affinity_pred_value column, ordered numerically where possible (M1, M2,
    ..., M10, ...) with any non-numeric names sorted alphabetically after them."""
    receptors = [m.group(1) for col in df.columns if (m := RECEPTOR_AFFINITY_PATTERN.match(col))]
    return sorted(receptors, key=_receptor_sort_key)


# --- Target matching --------------------------------------------------------------

# UniProt accession map for mAChR subtype target matching. Each accession carries its
# own "species" tag explicitly -- "human" for the human ortholog, a species name where
# known (e.g. "rat"), or None where the dataset doesn't establish which species it is.
# include_orthologs=False (see match_target()) matches on this tag, not on list
# position, so reordering or adding an entry can never silently change which one counts
# as "human". Counts below are each accession's total row count across
# Ki_uniprot/IC50_uniprot/EC50_uniprot combined in combined_results_with_anchor.csv,
# verified directly against that file -- re-check with the same kind of groupby if this
# map is ever edited.
#
# Every species tag below is deliberately None right now, including the entries
# previously confirmed as human/rat -- pending re-verification against UniProt before
# they're filled back in. While every tag is None, include_orthologs=False (and the
# CLI's --human-only) will match nothing for any receptor, since there's no longer an
# accession tagged species=="human" to restrict to.
RECEPTOR_UNIPROTS = {
    "M1": [
        {"accession": "P11229", "species": None},  # n=501
        {"accession": "P08482", "species": None},  # n=323
        {"accession": "Q8WMX0", "species": None},  # n=4
    ],
    "M2": [
        {"accession": "P08172", "species": None},  # n=211
        {"accession": "P10980", "species": None},  # n=26
        {"accession": "P06199", "species": None},  # n=24
    ],
    "M3": [
        {"accession": "P20309", "species": None},  # n=258
        {"accession": "P08483", "species": None},  # n=29
    ],
    "M4": [
        {"accession": "P08173", "species": None},  # n=305
        {"accession": "P08485", "species": None},  # n=25
    ],
    "M5": [
        {"accession": "P08912", "species": None},  # n=97
        {"accession": "P08911", "species": None},  # n=11
    ],
}

# Accessions deliberately NOT in RECEPTOR_UNIPROTS -- do not re-add:
#   P30542           - Adenosine A1 receptor, not muscarinic (e.g. CHEMBL4572905's Ki)
#   P11483, P17200   - do not appear in this dataset at all
# log_missing_ortholog_accessions() also flags Q8VH26/Q920H4/Q8VH27 (small n, muscarinic
# -named, not yet mapped to a subtype) -- left out deliberately pending confirmation of
# which subtype they actually belong to.

# Activity type -> (best-value column, target-uniprot column, target-name column).
# Only the "_y" value column is ever read -- see the module docstring for why.
ACTIVITY_COLUMNS = {
    "Ki": ("best_ki_nm_y", "Ki_uniprot", "Ki_target_name"),
    "IC50": ("best_ic50_nm_y", "IC50_uniprot", "IC50_target_name"),
    "EC50": ("best_ec50_nm_y", "EC50_uniprot", "EC50_target_name"),
}

_SUBTYPE_TOKEN_PATTERN = re.compile(r'\bM[1-5]\b')


def _ambiguous_subtype_mask(target_name: pd.Series) -> pd.Series:
    """True where target_name mentions 2+ distinct mAChR subtype labels (e.g.
    "Muscarinic acetylcholine receptors; M1 & M2") -- ChEMBL's assay-target
    annotation is genuinely ambiguous for these rows when there's no uniprot to
    fall back on (see match_target())."""
    return target_name.apply(lambda name: len(set(_SUBTYPE_TOKEN_PATTERN.findall(name))) >= 2)


def match_target(df: pd.DataFrame, receptor: str, activity: str, target_map: dict = None,
                  include_orthologs: bool = True, excluded_uniprots=None) -> pd.Series:
    """Boolean mask over df: which rows' `<activity>_uniprot` (or, only when that's
    null, a substring match on `<activity>_target_name`) identifies `receptor` for
    this one activity type. Matching is independent per activity type -- a compound
    can be Ki-matched to one subtype and IC50-matched to a different one.

    An ambiguous target_name (naming 2+ mAChR subtypes, e.g. "...; M1 & M2") only
    disqualifies the name-fallback path. When a specific uniprot is also given, it's
    trusted over the looser name string: 38 real rows have exactly this ambiguous
    IC50_target_name with a clean single IC50_uniprot=P11229, and this module's own
    row-count sanity check (see test_correlate_data.py) only holds for M1/IC50 if
    those 38 rows are kept, not excluded.

    include_orthologs=False restricts matching to only target_map[receptor] entries
    explicitly tagged species=="human" -- a coarse human-only/all-orthologs toggle
    that reads each entry's own tag rather than assuming a fixed list position.
    excluded_uniprots (an optional set of accession strings, e.g. {"P08482"}) is a
    finer-grained knob layered on top of that: any of those accessions are removed
    from the accepted set regardless of include_orthologs, so a caller can keep every
    ortholog except one specific species, not just an all-or-human-only choice. Both
    only ever narrow the uniprot-based match; the name-fallback path (used only when a
    row's uniprot is null) can't be split by species since target_name text doesn't
    encode it."""
    target_map = target_map or RECEPTOR_UNIPROTS
    value_col, uniprot_col, name_col = ACTIVITY_COLUMNS[activity]
    entries = target_map.get(receptor, [])
    if include_orthologs:
        accepted = {entry["accession"] for entry in entries}
    else:
        accepted = {entry["accession"] for entry in entries if entry.get("species") == "human"}
    if excluded_uniprots:
        accepted -= set(excluded_uniprots)

    target_name = df[name_col].fillna('')
    uniprot = df[uniprot_col]
    has_uniprot = uniprot.notna()

    uniprot_match = uniprot.isin(accepted)
    receptor_pattern = re.compile(rf'\b{re.escape(receptor)}\b')
    name_match = target_name.apply(lambda name: bool(receptor_pattern.search(name)))
    ambiguous = _ambiguous_subtype_mask(target_name)

    matched = (has_uniprot & uniprot_match) | (~has_uniprot & name_match & ~ambiguous)

    logger.info(
        f"{receptor} / {activity}: {int((matched & has_uniprot).sum())} matched by uniprot, "
        f"{int((matched & ~has_uniprot).sum())} matched by name fallback "
        f"({int((~has_uniprot & ambiguous & name_match).sum())} ambiguous-named row(s) excluded "
        f"from the fallback)"
    )
    return matched


def log_missing_ortholog_accessions(df: pd.DataFrame, target_map: dict = None) -> None:
    """Logs any uniprot accession that appears in a `<activity>_uniprot` column whose
    `<activity>_target_name` contains "uscarinic" but isn't listed anywhere in
    target_map, so a genuinely missing mAChR ortholog surfaces instead of being
    silently dropped from every receptor's series."""
    target_map = target_map or RECEPTOR_UNIPROTS
    known = {entry["accession"] for entries in target_map.values() for entry in entries}
    for activity, (_, uniprot_col, name_col) in ACTIVITY_COLUMNS.items():
        musc = df[name_col].fillna('').str.contains('uscarinic', regex=False)
        missing = df.loc[musc & df[uniprot_col].notna() & ~df[uniprot_col].isin(known), uniprot_col]
        for accession, count in missing.value_counts().items():
            logger.warning(
                f"{activity}: uniprot '{accession}' appears in {count} muscarinic-named row(s) "
                f"but is not in RECEPTOR_UNIPROTS -- check whether it's a missing ortholog"
            )


def load_target_map(path: str = None) -> dict:
    """Loads a --target-map override JSON file, in the same shape as RECEPTOR_UNIPROTS
    ({"M1": [{"accession": "P11229", "species": "human"}, ...], ...} -- species may be
    null where unknown), or returns RECEPTOR_UNIPROTS unchanged if no path is given."""
    if not path:
        return RECEPTOR_UNIPROTS
    with open(path) as f:
        return json.load(f)


# --- p-scale conversion -----------------------------------------------------------

def pred_to_p(affinity_pred_value):
    """Converts Boltz-2's affinity_pred_value (assumed log10(IC50 in uM); more
    negative = more potent) to a pIC50-equivalent p-scale value: 6.0 -
    affinity_pred_value. If Boltz-2's output convention ever changes, this is the
    one line to edit."""
    return 6.0 - affinity_pred_value


def nm_to_p(value_nm):
    """Converts raw potency (Ki/IC50/EC50) in nM to its p-scale equivalent:
    9.0 - log10(value_nM). Works on a scalar, a pandas Series, or an array-like.
    Non-positive or missing input has no defined log and returns NaN, never +/-inf."""
    if isinstance(value_nm, pd.Series):
        safe = value_nm.where(value_nm > 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            return 9.0 - np.log10(safe)
    arr = np.asarray(value_nm, dtype=float)
    safe = np.where(arr > 0, arr, np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = 9.0 - np.log10(safe)
    return result.item() if result.ndim == 0 else result


ACTIVITY_P_LABEL = {"Ki": "pKi", "IC50": "pIC50", "EC50": "pEC50"}

# Both pred_to_p() and nm_to_p() compute the same quantity, -log10(concentration in
# M) -- predicted and experimental values are always on this one shared unit, which
# is why they're directly comparable on the same axes. Shown on both x/y axis labels
# so the unit is explicit on the figure itself, not just in this module's docstring.
P_SCALE_UNIT = "p-scale, -log10[M]"


# --- Data loading ------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Loads collate_combine_boltz2.py's anchor-merged output CSV. Warns if any
    best_*_nm_x column is present -- an upstream ChEMBL-merge artefact (see module
    docstring) this pipeline never reads, since the _y columns are the correct "best"
    values -- and logs any muscarinic-named accession missing from RECEPTOR_UNIPROTS."""
    df = pd.read_csv(path)
    x_columns = [c for c in df.columns if c.startswith('best_') and c.endswith('_nm_x')]
    if x_columns:
        logger.warning(
            f"Input has {x_columns} -- these are an upstream ChEMBL-merge artefact and are "
            f"never read by this pipeline (the _y columns are the correct 'best' values, and "
            f"are always at least as potent); consider dropping them in collate_combine_boltz2.py"
        )
    log_missing_ortholog_accessions(df)
    return df


# --- Filtering (per receptor) ------------------------------------------------------

@dataclass
class FilterConfig:
    """One filter's worth of thresholds for a single receptor+activity selection.
    Every threshold is a min/max pair and defaults to "off" (None); apply_filters()
    only applies a bound that's set. Distance is observed to range ~6.6-19.0 A in
    this dataset -- the orthosteric-vs-extracellular-vestibule boundary is
    scientifically interesting around 8-12 A, but that's a UI hint, not a hardcoded
    default."""
    max_distance: Optional[float] = None
    min_distance: Optional[float] = None
    min_confidence: Optional[float] = None
    max_confidence: Optional[float] = None
    min_probability: Optional[float] = None
    max_probability: Optional[float] = None


@dataclass
class FilterReport:
    """How many paired points apply_filters started with, how many each active
    filter removed (in the order applied), and how many remain. `as_text()` renders
    this as the compact block the GUI/log show so a low final n is always
    traceable to a specific filter."""
    receptor: str
    activity: str
    initial_n: int
    steps: list = field(default_factory=list)  # [(label, n_removed, n_remaining), ...]
    final_n: int = 0

    def as_text(self) -> str:
        lines = [f"{self.receptor} / {self.activity}: {self.initial_n} paired point(s) before filtering"]
        for label, removed, remaining in self.steps:
            lines.append(f"  {label}: -{removed} -> {remaining}")
        lines.append(f"Final n: {self.final_n}")
        return "\n".join(lines)


def apply_filters(df: pd.DataFrame, receptor: str, activity: str, config: FilterConfig,
                   target_map: dict = None, include_orthologs: bool = True, excluded_uniprots=None) -> tuple:
    """Builds receptor's target-matched, paired (both x and y present) rows for
    activity, then applies each active FilterConfig threshold in order, using only
    that receptor's own `<receptor>_*` columns -- a compound dropped for one
    receptor's bad pose can still appear in another receptor's plot. Returns
    (filtered_df, FilterReport). excluded_uniprots is passed straight through to
    match_target() -- see its docstring for the human-only vs. per-accession
    exclusion distinction."""
    value_col, _, _ = ACTIVITY_COLUMNS[activity]
    x_col = f"{receptor}_affinity_pred_value"
    dist_col = f"{receptor}_distance_to_orthosteric_site"
    conf_col = f"{receptor}_confidence_score"
    prob_col = f"{receptor}_affinity_probability_binary"

    matched = match_target(df, receptor, activity, target_map, include_orthologs, excluded_uniprots)
    paired = matched & df[x_col].notna() & df[value_col].notna()
    working = df.loc[paired].copy()
    initial_n = len(working)
    steps = []

    def _apply(label: str, mask: pd.Series) -> None:
        nonlocal working
        before = len(working)
        working = working[mask.reindex(working.index, fill_value=False)]
        removed = before - len(working)
        steps.append((label, removed, len(working)))

    if config.max_distance is not None:
        _apply(f"max distance to orthosteric site <= {config.max_distance} A", working[dist_col] <= config.max_distance)
    if config.min_distance is not None:
        _apply(f"min distance to orthosteric site >= {config.min_distance} A", working[dist_col] >= config.min_distance)
    if config.min_confidence is not None:
        _apply(f"min confidence score >= {config.min_confidence}", working[conf_col] >= config.min_confidence)
    if config.max_confidence is not None:
        _apply(f"max confidence score <= {config.max_confidence}", working[conf_col] <= config.max_confidence)
    if config.min_probability is not None:
        _apply(f"min binding probability >= {config.min_probability}", working[prob_col] >= config.min_probability)
    if config.max_probability is not None:
        _apply(f"max binding probability <= {config.max_probability}", working[prob_col] <= config.max_probability)

    report = FilterReport(receptor=receptor, activity=activity, initial_n=initial_n, steps=steps, final_n=len(working))
    return working, report


# --- Regression ----------------------------------------------------------------

@dataclass
class RegressionResult:
    slope: float
    intercept: float
    r_value: float
    r_squared: float
    p_value: float
    n: int
    method: str = "ols"  # "ols" or "wls" -- which fit produced these stats
    spearman_rho: Optional[float] = None
    spearman_p: Optional[float] = None
    # The underlying statsmodels fit, kept around only so _confidence_band() can ask
    # it for a correctly-weighted prediction interval without refitting. Not meant
    # for external callers (GUI/CLI) to use directly -- everything they need is
    # already unpacked into the plain fields above.
    sm_result: object = field(default=None, repr=False, compare=False)


def fit_regression(x, y, weights=None) -> Optional[RegressionResult]:
    """Fits x -> y via statsmodels: OLS when weights is None (today's default,
    numerically identical to the old scipy.stats.linregress-based implementation),
    or WLS when weights is given (e.g. each point's affinity_probability_binary --
    higher probability pulls the line harder). Also computes Spearman rho, which is
    always unweighted in both modes -- rank correlation has no single agreed-upon
    weighted extension, so this doesn't attempt to invent one.

    r_value (Pearson r) is derived as sign(slope) * sqrt(r_squared) rather than
    computed independently -- this identity holds for weighted simple regression
    exactly as it does for unweighted, so r_value**2 == r_squared always holds (no
    risk of the two silently disagreeing), and it's exactly how scipy.stats.linregress
    related its own rvalue to r**2 for the unweighted case this replaces.

    Returns None (never raises) if n < 2, x/y have zero variance (slope undefined),
    any weight is negative/NaN, every weight is zero (degenerate -- no point actually
    influences the fit), or statsmodels otherwise produces NaN parameters."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        return None
    if np.all(x == x[0]) or np.all(y == y[0]):
        return None

    method = "ols"
    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if len(w) != n or np.any(np.isnan(w)) or np.any(w < 0) or not np.any(w > 0):
            return None
        method = "wls"

    exog = sm.add_constant(x, has_constant='add')
    model = sm.WLS(y, exog, weights=w) if method == "wls" else sm.OLS(y, exog)
    sm_result = model.fit()

    if np.any(np.isnan(sm_result.params)):
        return None

    intercept, slope = sm_result.params
    r_squared = float(sm_result.rsquared)
    r_value = float(np.sign(slope) * np.sqrt(r_squared)) if r_squared > 0 else 0.0
    p_value = float(sm_result.pvalues[1])

    rho, rho_p = spearmanr(x, y)
    if np.isnan(rho):
        rho, rho_p = None, None

    return RegressionResult(
        slope=float(slope), intercept=float(intercept), r_value=r_value,
        r_squared=r_squared, p_value=p_value, n=n, method=method,
        spearman_rho=rho, spearman_p=rho_p, sm_result=sm_result,
    )


def _confidence_band(result: RegressionResult, x_fit: np.ndarray):
    """95% confidence band for the fitted line's mean response at each x_fit point,
    via the underlying statsmodels fit's own prediction-interval machinery -- correct
    for both OLS and WLS (a WLS band correctly narrows where weights/confidence are
    higher, which the old hand-rolled OLS-only standard-error formula this replaced
    couldn't express). Returns (lower, upper) arrays, or None if the fit's covariance
    is singular (e.g. n <= 2) or otherwise can't produce an interval."""
    if result.sm_result is None or result.n <= 2:
        return None
    exog_fit = sm.add_constant(x_fit, has_constant='add')
    try:
        ci = result.sm_result.get_prediction(exog_fit).conf_int(alpha=0.05)
    except Exception:
        return None
    return ci[:, 0], ci[:, 1]


# --- Plotting ----------------------------------------------------------------------

# Scatter marker size range for weighted-regression mode (see _marker_sizes()) --
# unweighted mode keeps the flat size below unchanged.
UNWEIGHTED_MARKER_SIZE = 45
MIN_WEIGHTED_MARKER_SIZE = 15
MAX_WEIGHTED_MARKER_SIZE = 120


def _marker_sizes(probability) -> np.ndarray:
    """Maps affinity_probability_binary (0-1) onto a scatter marker size in
    [MIN_WEIGHTED_MARKER_SIZE, MAX_WEIGHTED_MARKER_SIZE], linearly -- used only when
    weighted regression is on, so a bigger dot visually means "this point pulled the
    WLS line harder". NaN/missing probability maps to the minimum size, matching how
    fit_regression() treats a missing weight as 0 (no influence on the fit either)."""
    prob = np.nan_to_num(np.clip(np.asarray(probability, dtype=float), 0.0, 1.0), nan=0.0)
    return MIN_WEIGHTED_MARKER_SIZE + (MAX_WEIGHTED_MARKER_SIZE - MIN_WEIGHTED_MARKER_SIZE) * prob


# Chart chrome (colorblind-safe palette, dataviz skill defaults)
CHART_SURFACE = '#fcfcfb'
GRIDLINE_COLOR = '#e1e0d9'
AXIS_COLOR = '#c3c2b7'
MUTED_TEXT_COLOR = '#898781'
PRIMARY_TEXT_COLOR = '#0b0b0b'

# Fixed, colorblind-safe per-receptor colors -- consistent across every figure and
# legend, assigned by receptor name so M2 is always the same color.
RECEPTOR_COLORS = {
    "M1": "#2a78d6",  # blue
    "M2": "#d1495b",  # red
    "M3": "#2e8b57",  # green
    "M4": "#e08b1f",  # amber
    "M5": "#7b4fa8",  # purple
}
# Fallback palette for receptors outside RECEPTOR_COLORS, assigned deterministically
# by hashing the receptor name so color assignment doesn't depend on enumeration order.
_FALLBACK_PALETTE = ["#2a78d6", "#d1495b", "#2e8b57", "#e08b1f", "#7b4fa8", "#1f9aa3", "#c2790f", "#6b4fa0"]


def _receptor_color(receptor: str) -> str:
    if receptor in RECEPTOR_COLORS:
        return RECEPTOR_COLORS[receptor]
    digest = hashlib.md5(receptor.encode()).hexdigest()
    return _FALLBACK_PALETTE[int(digest, 16) % len(_FALLBACK_PALETTE)]


def _describe_filters(config: FilterConfig) -> str:
    """One-line, human-readable summary of a FilterConfig's active thresholds, for
    the figure title's provenance line."""
    parts = []
    if config.max_distance is not None:
        parts.append(f"dist<={config.max_distance}A")
    if config.min_distance is not None:
        parts.append(f"dist>={config.min_distance}A")
    if config.min_confidence is not None:
        parts.append(f"conf>={config.min_confidence}")
    if config.max_confidence is not None:
        parts.append(f"conf<={config.max_confidence}")
    if config.min_probability is not None:
        parts.append(f"prob>={config.min_probability}")
    if config.max_probability is not None:
        parts.append(f"prob<={config.max_probability}")
    return ", ".join(parts)


def _paired_p_scale(filtered: pd.DataFrame, receptor: str, activity: str) -> pd.DataFrame:
    """Converts a receptor/activity's filtered rows to p-scale x/y pairs, dropping
    any row where either conversion is undefined (e.g. a non-positive nM value).
    Carries along everything the GUI's click-to-inspect compound panel needs
    (canonical_smiles, molecular_weight, alogp, per-receptor distance/confidence/
    probability, raw experimental value, matched target name + uniprot) so a clicked scatter
    point's row can be looked up directly from this DataFrame with no extra query --
    columns are filled with None if the input CSV doesn't have them, rather than
    raising, since only the p-scale x/y pair is actually required for plotting."""
    x_col = f"{receptor}_affinity_pred_value"
    dist_col = f"{receptor}_distance_to_orthosteric_site"
    conf_col = f"{receptor}_confidence_score"
    prob_col = f"{receptor}_affinity_probability_binary"
    value_col, uniprot_col, name_col = ACTIVITY_COLUMNS[activity]

    def _col(name: str):
        return filtered[name].values if name in filtered.columns else None

    pair_df = pd.DataFrame({
        "chembl_id": filtered["chembl_id"].values,
        "pred_p": np.asarray(pred_to_p(filtered[x_col]), dtype=float),
        "exp_p": np.asarray(nm_to_p(filtered[value_col]), dtype=float),
        "experimental_value_nm": _col(value_col),
        "distance_to_orthosteric_site": _col(dist_col),
        "confidence_score": _col(conf_col),
        "affinity_probability_binary": _col(prob_col),
        "canonical_smiles": _col("canonical_smiles"),
        "molecular_weight": _col("molecular_weight"),
        "alogp": _col("alogp"),
        "target_name": _col(name_col),
        "target_uniprot": _col(uniprot_col),
    })
    return pair_df.dropna(subset=["pred_p", "exp_p"])


def make_figure(df: pd.DataFrame, receptors, activity: str, config: FilterConfig,
                layout: str = 'overlay', min_points: int = 4, annotate_points: bool = False,
                target_map: dict = None, include_orthologs: bool = True, excluded_uniprots=None,
                weighted: bool = False, figsize_per_panel: tuple = (6, 6)) -> tuple:
    """Builds one Figure for the given receptors x activity selection -- layout
    'overlay' (all receptors on one Axes, sharing one pooled regression line/R2 fit
    across every selected receptor's points -- points stay color-coded per receptor,
    only the fit itself is combined) or 'facet' (one subplot per receptor, each with
    its own independent regression). Both axes are always p-scale; never nM (see
    module docstring). weighted=True switches every fit (per-receptor in facet mode,
    the single pooled fit in overlay mode) from OLS to WLS weighted by each point's
    affinity_probability_binary, and scales each scatter point's size by that same
    probability (see _marker_sizes()) -- weighted=False (default) is numerically
    identical to this function's behavior before weighted regression existed. Returns
    (fig, {receptor: FilterReport}, {receptor: pair_df}) -- the third element is the
    same per-receptor DataFrame the scatter points were plotted from (chembl_id/
    pred_p/exp_p plus compound info -- see _paired_p_scale()), so a caller with a GUI
    event (e.g. a matplotlib pick_event) can look up a clicked point's full row with
    no extra query: each scatter artist's gid is set to its receptor name via
    set_gid(), and event.ind indexes into that receptor's pair_df in the same order
    the points were plotted. Never calls plt.show() or attaches to pyplot's global
    figure registry -- callers (CLI: fig.savefig(); GUI: FigureCanvasQTAgg(fig)) own
    the figure's lifecycle."""
    if isinstance(receptors, str):
        receptors = [receptors]
    if not receptors:
        raise ValueError("make_figure requires at least one receptor")

    filter_reports = {}
    pair_data = {}
    for receptor in receptors:
        filtered, report = apply_filters(df, receptor, activity, config, target_map, include_orthologs, excluded_uniprots)
        filter_reports[receptor] = report
        pair_data[receptor] = _paired_p_scale(filtered, receptor, activity)

    n_panels = len(receptors) if layout == 'facet' else 1
    fig = Figure(figsize=(figsize_per_panel[0] * n_panels, figsize_per_panel[1]))
    axes = fig.subplots(1, n_panels, squeeze=False)[0]
    activity_label = ACTIVITY_P_LABEL[activity]

    # In overlay mode every receptor shares one Axes, so anything placed at a fixed
    # axes-fraction position (annotations, "no data" text) must be stacked instead of
    # each receptor independently claiming the same spot -- otherwise two receptors'
    # text (or two "no data" labels) land exactly on top of each other. facet mode
    # doesn't need this: each receptor already has its own dedicated Axes.
    any_plotted = False
    no_data_receptors = []
    axes_with_data = set()
    combined_pred = []
    combined_exp = []
    combined_weights = []

    for i, receptor in enumerate(receptors):
        ax = axes[0] if layout == 'overlay' else axes[i]
        pair_df = pair_data[receptor]
        n = len(pair_df)
        color = _receptor_color(receptor)

        if n > 0:
            any_plotted = True
            axes_with_data.add(ax)
            x = pair_df["pred_p"].values
            y = pair_df["exp_p"].values
            probability = pair_df["affinity_probability_binary"].values
            sizes = _marker_sizes(probability) if weighted else UNWEIGHTED_MARKER_SIZE
            scatter = ax.scatter(x, y, s=sizes, color=color, edgecolors=CHART_SURFACE, linewidths=1,
                                  alpha=0.9, label=receptor, zorder=3, picker=True)
            scatter.set_pickradius(6)
            scatter.set_gid(receptor)

            if annotate_points:
                for chembl_id, xi, yi in zip(pair_df["chembl_id"], x, y):
                    ax.annotate(chembl_id, (xi, yi), fontsize=6, color=MUTED_TEXT_COLOR,
                                 xytext=(3, 3), textcoords='offset points')

            if layout == 'overlay':
                # Overlay draws one pooled regression across every selected receptor
                # after this loop (see below) rather than one line per receptor --
                # collect this receptor's points into the combined series instead of
                # fitting here.
                combined_pred.append(x)
                combined_exp.append(y)
                combined_weights.append(np.nan_to_num(probability, nan=0.0))
            else:
                weights = np.nan_to_num(probability, nan=0.0) if weighted else None
                result = fit_regression(x, y, weights=weights) if n >= min_points else None

                if result is None:
                    ax.text(0.05, 0.95, f"insufficient data (n={n})", transform=ax.transAxes,
                             ha='left', va='top', color=MUTED_TEXT_COLOR, fontsize=9)
                    if n >= min_points:
                        logger.warning(f"{receptor} / {activity}: regression degenerate (zero variance) despite n={n}")
                    else:
                        logger.info(f"{receptor} / {activity}: n={n} < min_points={min_points}, no regression drawn")
                else:
                    x_fit = np.linspace(x.min(), x.max(), 50)
                    y_fit = result.slope * x_fit + result.intercept
                    ax.plot(x_fit, y_fit, color=color, linewidth=1.5, alpha=0.85, zorder=2)

                    band = _confidence_band(result, x_fit)
                    if band is not None:
                        ax.fill_between(x_fit, band[0], band[1], color=color, alpha=0.15, zorder=1, linewidth=0)

                    small_n_marker = "[small n] " if n < 10 else ""
                    rho_text = f", Spearman rho={result.spearman_rho:.2f}" if result.spearman_rho is not None else ""
                    annotation = (
                        f"y = {result.slope:.2f}x + {result.intercept:.2f}  [{result.method.upper()}]\n"
                        f"{small_n_marker}$R^2$={result.r_squared:.2f}, r={result.r_value:.2f}, "
                        f"p={result.p_value:.3f}, n={n}{rho_text}"
                    )
                    ax.text(0.05, 0.95, annotation, transform=ax.transAxes, ha='left', va='top',
                             color=PRIMARY_TEXT_COLOR, fontsize=9)
                    if n < 10:
                        logger.warning(f"{receptor} / {activity}: small n ({n}) -- R2 is not a validation result")
                    logger.info(
                        f"{receptor} / {activity}: {result.method.upper()} slope={result.slope:.3f}, "
                        f"R2={result.r_squared:.3f}, Pearson r={result.r_value:.3f} (p={result.p_value:.4f}), "
                        f"Spearman rho={result.spearman_rho}, n={n}"
                    )
        elif layout == 'facet':
            # Safe to write directly to this panel's own Axes -- no other receptor
            # shares it.
            ax.text(0.5, 0.5, f"{receptor}: no data", ha='center', va='center',
                     color=MUTED_TEXT_COLOR, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            no_data_receptors.append(receptor)

        # Chart chrome applies unconditionally, including "no data" panels, so every
        # panel in a facet looks consistent regardless of which receptors have data.
        # Labels/title are per-receptor here only in facet mode -- in overlay mode
        # every receptor shares this Axes, so setting a per-receptor label inside this
        # loop would just have the last-processed receptor silently overwrite it (a
        # real bug this replaced: the x-axis used to end up labeled with whichever
        # receptor came last, e.g. "M5", even though M1-M4 were also plotted on it).
        ax.set_facecolor(CHART_SURFACE)
        if layout == 'facet':
            ax.set_xlabel(f"{receptor} predicted affinity ({P_SCALE_UNIT})", color=MUTED_TEXT_COLOR)
            ax.set_ylabel(f"Experimental {activity_label} ({P_SCALE_UNIT})", color=MUTED_TEXT_COLOR)
            ax.set_title(receptor, color=PRIMARY_TEXT_COLOR, fontsize=11)
        ax.grid(True, color=GRIDLINE_COLOR, linewidth=1)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(AXIS_COLOR)
        ax.tick_params(colors=MUTED_TEXT_COLOR)

    if layout == 'overlay':
        axes[0].set_xlabel(f"Predicted affinity ({P_SCALE_UNIT})", color=MUTED_TEXT_COLOR)
        axes[0].set_ylabel(f"Experimental {activity_label} ({P_SCALE_UNIT})", color=MUTED_TEXT_COLOR)

        if any_plotted:
            # One pooled regression across every selected receptor's points -- not
            # one line/R2 per receptor. Points stay color-coded per receptor (see the
            # legend) but the fit itself treats every selected receptor's data as a
            # single combined series.
            x_all = np.concatenate(combined_pred)
            y_all = np.concatenate(combined_exp)
            n_total = len(x_all)
            weights_all = np.concatenate(combined_weights) if weighted else None
            result = fit_regression(x_all, y_all, weights=weights_all) if n_total >= min_points else None

            if result is None:
                axes[0].text(0.05, 0.95, f"insufficient data (n={n_total})", transform=axes[0].transAxes,
                             ha='left', va='top', color=MUTED_TEXT_COLOR, fontsize=9)
                if n_total >= min_points:
                    logger.warning(f"{activity}: combined regression degenerate (zero variance) despite n={n_total}")
                else:
                    logger.info(f"{activity}: combined n={n_total} < min_points={min_points}, no regression drawn")
            else:
                x_fit = np.linspace(x_all.min(), x_all.max(), 50)
                y_fit = result.slope * x_fit + result.intercept
                axes[0].plot(x_fit, y_fit, color=PRIMARY_TEXT_COLOR, linewidth=1.5, alpha=0.85, zorder=2)

                band = _confidence_band(result, x_fit)
                if band is not None:
                    axes[0].fill_between(x_fit, band[0], band[1], color=PRIMARY_TEXT_COLOR, alpha=0.12, zorder=1, linewidth=0)

                small_n_marker = "[small n] " if n_total < 10 else ""
                rho_text = f", Spearman rho={result.spearman_rho:.2f}" if result.spearman_rho is not None else ""
                annotation = (
                    f"y = {result.slope:.2f}x + {result.intercept:.2f}  [{result.method.upper()}]\n"
                    f"{small_n_marker}$R^2$={result.r_squared:.2f}, r={result.r_value:.2f}, "
                    f"p={result.p_value:.3f}, n={n_total}{rho_text}"
                )
                axes[0].text(0.05, 0.95, annotation, transform=axes[0].transAxes, ha='left', va='top',
                             color=PRIMARY_TEXT_COLOR, fontsize=9)
                if n_total < 10:
                    logger.warning(f"{activity}: small combined n ({n_total}) -- R2 is not a validation result")
                plotted_receptors = [r for r in receptors if r not in no_data_receptors]
                logger.info(
                    f"{activity}: combined {result.method.upper()} across {plotted_receptors} slope={result.slope:.3f}, "
                    f"R2={result.r_squared:.3f}, Pearson r={result.r_value:.3f} (p={result.p_value:.4f}), "
                    f"Spearman rho={result.spearman_rho}, n={n_total}"
                )

            if no_data_receptors:
                # Some receptors plotted, some didn't -- top-right corner, since the
                # legend (if any) defaults to lower-right and the regression
                # annotation sits at top-left.
                axes[0].text(0.99, 0.98, f"no data: {', '.join(no_data_receptors)}", transform=axes[0].transAxes,
                             ha='right', va='top', color=MUTED_TEXT_COLOR, fontsize=8)
        else:
            # Every selected receptor is empty -- one combined message, ticks cleared
            # since there's nothing to show a scale for.
            axes[0].text(0.5, 0.5, f"no data: {', '.join(no_data_receptors)}", ha='center', va='center',
                         color=MUTED_TEXT_COLOR, transform=axes[0].transAxes)
            axes[0].set_xticks([])
            axes[0].set_yticks([])

    # Identity line: drawn once per Axes that actually has data, after every receptor
    # sharing that Axes has been plotted -- not once per receptor, which in overlay
    # mode would redraw a slightly-shifted dashed line on every iteration as the
    # shared Axes' limits grow. Both x and y are set to the same combined range
    # (rather than each axis keeping its own independently-scaled range) so the
    # identity line runs at a true 45 degrees and predicted vs experimental values
    # are directly comparable on a like-for-like scale.
    for ax in axes_with_data:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        identity_lo, identity_hi = min(xlim[0], ylim[0]), max(xlim[1], ylim[1])
        ax.plot([identity_lo, identity_hi], [identity_lo, identity_hi], linestyle='--',
                 color=AXIS_COLOR, linewidth=1, zorder=0)
        ax.set_xlim(identity_lo, identity_hi)
        ax.set_ylim(identity_lo, identity_hi)

    # Every panel -- with or without data -- gets a square, equal-scale box, so one
    # unit on the x axis is always the same physical length as one unit on the y axis.
    for ax in axes:
        ax.set_aspect('equal', adjustable='box')

    if layout == 'overlay' and len(receptors) > 1 and any_plotted:
        axes[0].legend(loc='lower right', fontsize=8, facecolor=CHART_SURFACE, edgecolor=AXIS_COLOR)

    desc_parts = []
    filters_desc = _describe_filters(config)
    if filters_desc:
        desc_parts.append(filters_desc)
    if excluded_uniprots:
        desc_parts.append(f"excl uniprot: {', '.join(sorted(excluded_uniprots))}")
    if weighted:
        desc_parts.append("weighted (WLS)")
    title = f"{', '.join(receptors)}: predicted vs experimental {activity_label}"
    if desc_parts:
        title += f" ({'; '.join(desc_parts)})"
    fig.suptitle(title, color=PRIMARY_TEXT_COLOR, fontsize=13)
    fig.tight_layout()

    return fig, filter_reports, pair_data


# --- Distribution histograms -----------------------------------------------------

# (display label, pair_df column name) -- every numeric column _paired_p_scale()
# carries, selectable as a histogram's x-axis in the GUI's Distributions tab (and
# usable directly via make_histogram() from a script/CLI too).
HISTOGRAM_FIELDS = [
    ("Predicted affinity (p-scale)", "pred_p"),
    ("Experimental affinity (p-scale)", "exp_p"),
    ("Distance to orthosteric site (A)", "distance_to_orthosteric_site"),
    ("Confidence score", "confidence_score"),
    ("Binding probability", "affinity_probability_binary"),
    ("Molecular weight", "molecular_weight"),
    ("AlogP", "alogp"),
]


def make_histogram(pair_data: dict, field: str, field_label: str = None, bins: int = 20,
                    figsize: tuple = (6, 5)) -> Figure:
    """Builds a frequency histogram of `field` (a column in each receptor's pair_df --
    see HISTOGRAM_FIELDS for the standard set, or _paired_p_scale() for the full
    column list) across every receptor in pair_data -- the same {receptor: pair_df}
    mapping make_figure() returns as its third value, so the histogram always
    reflects exactly the same filtered/matched points the correlation plot is built
    from. Each receptor's values are stacked in its own color (see RECEPTOR_COLORS)
    within one shared set of bins, so a receptor with unusually many or few points is
    visually obvious. Rows where `field` is NaN/missing are dropped per receptor
    (e.g. not every compound has a confidence_score). Returns a bare Figure (no
    canvas, no pyplot), same lifecycle contract as make_figure() -- callers own it."""
    if field_label is None:
        field_label = field
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)

    receptor_values = {}
    for receptor, pair_df in pair_data.items():
        if field not in pair_df.columns:
            continue
        values = pd.to_numeric(pair_df[field], errors='coerce').dropna().values
        if len(values) > 0:
            receptor_values[receptor] = values

    if not receptor_values:
        ax.text(0.5, 0.5, "no data", ha='center', va='center',
                 color=MUTED_TEXT_COLOR, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        combined = np.concatenate(list(receptor_values.values()))
        bin_edges = np.histogram_bin_edges(combined, bins=bins)
        colors = [_receptor_color(r) for r in receptor_values]
        ax.hist(list(receptor_values.values()), bins=bin_edges, stacked=True,
                 color=colors, edgecolor=CHART_SURFACE, linewidth=0.5,
                 label=list(receptor_values.keys()), zorder=3)
        if len(receptor_values) > 1:
            ax.legend(loc='upper right', fontsize=8, facecolor=CHART_SURFACE, edgecolor=AXIS_COLOR)

    ax.set_xlabel(field_label, color=MUTED_TEXT_COLOR)
    ax.set_ylabel("Frequency", color=MUTED_TEXT_COLOR)
    ax.set_facecolor(CHART_SURFACE)
    ax.grid(True, color=GRIDLINE_COLOR, linewidth=1, axis='y', zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(AXIS_COLOR)
    ax.tick_params(colors=MUTED_TEXT_COLOR)
    fig.suptitle(f"Distribution: {field_label}", color=PRIMARY_TEXT_COLOR, fontsize=13)
    fig.tight_layout()
    return fig


# --- Export --------------------------------------------------------------------

def save_figure(fig: Figure, path: str, dpi: int = 300, transparent: bool = False) -> None:
    """Saves fig in whatever format `path`'s extension implies (png/svg/pdf/tiff/...).
    dpi is ignored for the two vector formats (svg, pdf) -- matplotlib silently
    accepts (and no-ops) the kwarg there, which is what we want rather than pretending
    a vector format has a fixed resolution."""
    facecolor = 'none' if transparent else CHART_SURFACE
    fig.savefig(path, dpi=dpi, facecolor=facecolor, transparent=transparent)
    logger.info(f"Saved figure to '{path}'")


def _paired_export_rows(df: pd.DataFrame, receptor: str, activity: str, config: FilterConfig,
                         target_map: dict = None, include_orthologs: bool = True,
                         excluded_uniprots=None) -> pd.DataFrame:
    """Builds the exact paired rows behind one receptor/activity's figure -- chembl_id,
    receptor, activity, raw + p-scale x/y, and the filter columns that were checked --
    so the figure is reproducible from the exported table alone. Column names avoid
    the literal substring "nM" even for the raw-nanomolar column, so a stray header
    can never be misread as claiming the (always p-scale) plotted axes are in nM."""
    filtered, _ = apply_filters(df, receptor, activity, config, target_map, include_orthologs, excluded_uniprots)
    x_col = f"{receptor}_affinity_pred_value"
    dist_col = f"{receptor}_distance_to_orthosteric_site"
    conf_col = f"{receptor}_confidence_score"
    prob_col = f"{receptor}_affinity_probability_binary"
    value_col = ACTIVITY_COLUMNS[activity][0]

    export_df = pd.DataFrame({
        "chembl_id": filtered["chembl_id"].values,
        "receptor": receptor,
        "activity": activity,
        "predicted_affinity_pred_value": filtered[x_col].values,
        "predicted_p": np.asarray(pred_to_p(filtered[x_col]), dtype=float),
        "experimental_value_nanomolar": filtered[value_col].values,
        "experimental_p": np.asarray(nm_to_p(filtered[value_col]), dtype=float),
        "distance_to_orthosteric_site": filtered[dist_col].values,
        "confidence_score": filtered[conf_col].values,
        "affinity_probability_binary": filtered[prob_col].values,
        "molecular_weight": filtered["molecular_weight"].values,
        "alogp": filtered["alogp"].values,
    })
    return export_df.dropna(subset=["predicted_p", "experimental_p"])


def export_paired_csv(df: pd.DataFrame, receptors, activity: str, config: FilterConfig,
                       path: str, target_map: dict = None, include_orthologs: bool = True,
                       excluded_uniprots=None) -> pd.DataFrame:
    """Exports the exact paired points behind a figure to `path` -- one or more
    receptors' rows (see _paired_export_rows), concatenated, so a multi-receptor
    overlay/facet figure exports as a single reproducible table."""
    if isinstance(receptors, str):
        receptors = [receptors]
    export_df = pd.concat(
        [_paired_export_rows(df, r, activity, config, target_map, include_orthologs, excluded_uniprots) for r in receptors],
        ignore_index=True,
    )
    export_df.to_csv(path, index=False)
    logger.info(f"Exported {len(export_df)} paired point(s) across {receptors} to '{path}'")
    return export_df


# --- CLI -----------------------------------------------------------------------

def parse_arguments():
    """Parse command line arguments for configuration options."""
    parser = argparse.ArgumentParser(
        description="Correlate Boltz-2 predicted affinity against experimental IC50/EC50/Ki, per receptor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input',
        type=str,
        default='combined_results_with_anchor.csv',
        help='collate_combine_boltz2.py --anchor-csv output CSV'
    )
    parser.add_argument(
        '--receptors',
        type=str,
        default='all',
        help='Comma-separated receptor names to plot (e.g. "M1,M2"), or "all"'
    )
    parser.add_argument(
        '--activity',
        type=str,
        choices=list(ACTIVITY_COLUMNS),
        default='Ki',
        help='Which single activity type to plot per figure'
    )
    parser.add_argument(
        '--layout',
        type=str,
        choices=['overlay', 'facet'],
        default='overlay',
        help='overlay: all selected receptors on one axes; facet: one subplot per receptor'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Figure output path; format is inferred from the extension (.png/.svg/.pdf/.tiff/...). '
             'Defaults to "<activity>_<receptors>.png" in the current directory if omitted.'
    )
    parser.add_argument('--dpi', type=int, default=300, help='Export DPI (ignored for svg/pdf)')
    parser.add_argument('--transparent', action='store_true', help='Export with a transparent background')
    parser.add_argument('--annotate-points', action='store_true', help='Label each point with its chembl_id')
    parser.add_argument('--export-csv', type=str, default=None,
                         help='Optional path to also export the plotted pairs (across all selected receptors) as CSV')

    parser.add_argument('--target-map', type=str, default=None,
                         help='Optional JSON file overriding RECEPTOR_UNIPROTS, e.g. '
                              '{"M1": [{"accession": "P11229", "species": "human"}, ...], ...}')
    parser.add_argument('--include-orthologs', dest='include_orthologs', action='store_true', default=True,
                         help='Match non-human orthologs too (default)')
    parser.add_argument('--human-only', dest='include_orthologs', action='store_false',
                         help='Restrict target matching to each receptor\'s species=="human"-tagged accession(s) only')
    parser.add_argument('--exclude-uniprot', action='append', default=None, metavar='ACCESSION',
                         help='Exclude one UniProt accession from target matching (repeatable), e.g. '
                              '--exclude-uniprot P08482 -- finer-grained than --human-only, which drops '
                              'every non-human accession at once')
    parser.add_argument('--min-points', type=int, default=4,
                         help='Minimum paired points required to draw a regression line')
    parser.add_argument('--weighted-regression', action='store_true',
                         help='Fit WLS (weighted by each point\'s affinity_probability_binary) instead of '
                              'OLS, and scale each scatter point\'s size by that same probability')

    parser.add_argument('--max-distance', type=float, default=None, help='Filter: max distance to orthosteric site (A)')
    parser.add_argument('--min-distance', type=float, default=None, help='Filter: min distance to orthosteric site (A)')
    parser.add_argument('--min-confidence', type=float, default=None, help='Filter: min confidence_score')
    parser.add_argument('--max-confidence', type=float, default=None, help='Filter: max confidence_score')
    parser.add_argument('--min-probability', type=float, default=None, help='Filter: min affinity_probability_binary')
    parser.add_argument('--max-probability', type=float, default=None, help='Filter: max affinity_probability_binary')

    return parser.parse_args()


def _filter_config_from_args(args) -> FilterConfig:
    return FilterConfig(
        max_distance=args.max_distance, min_distance=args.min_distance,
        min_confidence=args.min_confidence, max_confidence=args.max_confidence,
        min_probability=args.min_probability, max_probability=args.max_probability,
    )


def main():
    """Main execution function: loads data, resolves the requested receptor(s) and
    activity type, builds one figure, saves it, and (optionally) exports the paired
    data as CSV."""
    args = parse_arguments()
    df = load_data(args.input)
    target_map = load_target_map(args.target_map)

    all_receptors = find_receptors(df)
    if not all_receptors:
        raise ValueError(f"No '<receptor>_affinity_pred_value' columns found in '{args.input}'")
    logger.info(f"Found {len(all_receptors)} receptors: {all_receptors}")

    if args.receptors == 'all':
        receptors = all_receptors
    else:
        receptors = [r.strip() for r in args.receptors.split(',') if r.strip()]
        unknown = [r for r in receptors if r not in all_receptors]
        if unknown:
            raise ValueError(f"Unknown receptor(s) {unknown} -- available: {all_receptors}")

    config = _filter_config_from_args(args)
    excluded_uniprots = set(args.exclude_uniprot) if args.exclude_uniprot else None
    fig, filter_reports, _ = make_figure(
        df, receptors, args.activity, config, layout=args.layout, min_points=args.min_points,
        annotate_points=args.annotate_points, target_map=target_map, include_orthologs=args.include_orthologs,
        excluded_uniprots=excluded_uniprots, weighted=args.weighted_regression,
    )
    for report in filter_reports.values():
        logger.info("\n" + report.as_text())

    output = args.output or f"{args.activity}_{'_'.join(receptors)}.png"
    save_figure(fig, output, dpi=args.dpi, transparent=args.transparent)

    if args.export_csv:
        export_paired_csv(df, receptors, args.activity, config, args.export_csv, target_map,
                           args.include_orthologs, excluded_uniprots)


if __name__ == "__main__":
    main()
