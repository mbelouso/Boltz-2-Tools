'''
Filters/reshapes ChEMBL search-results CSVs. Two independent modes, selected via a
subcommand:

keyword-filter
    Keeps rows whose full text (all columns concatenated, case-insensitive) matches
    every filter group in FILTERS below (each group is an OR of substrings).

    python filter_searched_results.py keyword-filter --input search_results.csv --output filtered_results.csv

    Current filters (a row must pass every one to be kept):
        1. "Muscarinic"
        2. One of: P11229, P08172, P20309, P08173, P08912
           (UniProt accessions for CHRM1-CHRM5)

subtype-pairs
    Takes a ChEMBL per-activity-type reference CSV, the raw pre-rename format with the
    merge "_y" suffix still on the value columns (chembl_id, best_ki_nm_y, Ki_uniprot,
    best_ic50_nm_y, IC50_uniprot, best_ec50_nm_y, EC50_uniprot, ... -- see
    collate_combine_boltz2.py's REFERENCE_COLUMNS_RENAME for the same column mapping) and, for
    each activity type (Ki / IC50 / EC50):
        1. takes compounds that have a value for that activity type (best_X_y not null)
        2. checks the target it was measured against (the X_uniprot column)
        3. keeps it only if that target is one of the 5 mAChR subtypes
           (P11229/P08172/P20309/P08173/P08912)
        4. relabels the target as M1-M5
    producing a wide table: chembl_id, M1_Ki_nm, M1_IC50_nm, M1_EC50_nm, M2_Ki_nm, ...,
    M5_EC50_nm. Compounds with no mAChR-subtype match for any activity type are dropped.

    python filter_searched_results.py subtype-pairs --input reference_results.csv --output subtype_pairs.csv
'''

import argparse
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- keyword-filter mode -----------------------------------------------------

# Each entry is a list of substrings; a row passes that filter if it contains at least
# one of them (case-insensitive). A row must pass every filter below to be kept.
FILTERS = [
    ['Muscarinic'],
    ['P11229', 'P08172', 'P20309', 'P08173', 'P08912'],
]


def row_matches_filter(row_text: str, substrings: list) -> bool:
    """Return True if row_text contains at least one of the given substrings."""
    return any(substring.lower() in row_text for substring in substrings)


def filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows that match every filter in FILTERS (each filter is an OR of
    substrings, checked against the row's full text across all columns)."""
    row_texts = df.astype(str).apply(lambda row: ' '.join(row.values).lower(), axis=1)

    keep_mask = pd.Series(True, index=df.index)
    for substrings in FILTERS:
        matches = row_texts.apply(lambda text: row_matches_filter(text, substrings))
        logger.info(f"Filter {substrings}: {matches.sum()}/{len(df)} rows match")
        keep_mask &= matches

    return df[keep_mask]


def run_keyword_filter(args):
    """Load --input, keep rows matching every FILTERS group, write to --output."""
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(
        f"{input_path.stem}_filtered{input_path.suffix}"
    )

    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows from '{input_path}'")

    filtered_df = filter_rows(df)
    logger.info(f"Kept {len(filtered_df)}/{len(df)} rows after filtering")

    filtered_df.to_csv(output_path, index=False)
    logger.info(f"Filtered results exported to '{output_path}'")


# --- subtype-pairs mode -------------------------------------------------------

# UniProt accession -> muscarinic receptor subtype label.
UNIPROT_TO_SUBTYPE = {
    'P11229': 'M1',
    'P08172': 'M2',
    'P20309': 'M3',
    'P08173': 'M4',
    'P08912': 'M5',
}
SUBTYPE_ORDER = ['M1', 'M2', 'M3', 'M4', 'M5']

# Activity type -> (best-value column, target-uniprot column) in the reference CSV.
# The value columns keep the raw "_y" merge-suffix (e.g. best_ic50_nm_y) as produced
# by the original ChEMBL retrieval pipeline -- see collate_combine_boltz2.py's
# REFERENCE_COLUMNS_RENAME for the same mapping.
ACTIVITY_COLUMNS = {
    'Ki': ('best_ki_nm_y', 'Ki_uniprot'),
    'IC50': ('best_ic50_nm_y', 'IC50_uniprot'),
    'EC50': ('best_ec50_nm_y', 'EC50_uniprot'),
}


def build_subtype_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """For each activity type, keep compounds whose best value was measured against one
    of the 5 mAChR subtypes, relabel the target as M1-M5, and lay the result out as a
    wide table (chembl_id, M1_Ki_nm, M1_IC50_nm, ..., M5_EC50_nm). Compounds with no
    mAChR-subtype match for any activity type are dropped."""
    new_columns = [f"{subtype}_{activity}_nm" for subtype in SUBTYPE_ORDER for activity in ACTIVITY_COLUMNS]
    result = df[['chembl_id']].copy()
    for column in new_columns:
        result[column] = pd.NA

    for activity, (value_col, uniprot_col) in ACTIVITY_COLUMNS.items():
        if value_col not in df.columns or uniprot_col not in df.columns:
            logger.warning(f"'{value_col}' or '{uniprot_col}' not found in input -- skipping {activity}")
            continue

        subtype_series = df[uniprot_col].map(UNIPROT_TO_SUBTYPE)
        valid = df[value_col].notna() & subtype_series.notna()
        logger.info(f"{activity}: {valid.sum()}/{len(df)} compounds have a best value against an mAChR subtype")

        for subtype in SUBTYPE_ORDER:
            mask = valid & (subtype_series == subtype)
            result.loc[mask, f"{subtype}_{activity}_nm"] = df.loc[mask, value_col]

    has_any_pair = result[new_columns].notna().any(axis=1)
    dropped = (~has_any_pair).sum()
    if dropped:
        logger.info(f"Dropping {dropped} compounds with no mAChR-subtype match for any activity type")

    return result[has_any_pair]


def run_subtype_pairs(args):
    """Load --input reference CSV, reshape to per-subtype Ki/IC50/EC50 columns, write to --output."""
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(
        f"{input_path.stem}_subtype_pairs{input_path.suffix}"
    )

    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows from '{input_path}'")

    if 'chembl_id' not in df.columns:
        raise ValueError(f"'{input_path}' has no chembl_id column")

    pairs_df = build_subtype_pairs(df)
    logger.info(f"Kept {len(pairs_df)}/{len(df)} compounds with at least one mAChR-subtype pairing")

    pairs_df.to_csv(output_path, index=False)
    logger.info(f"Subtype pairs exported to '{output_path}'")


# --- CLI -----------------------------------------------------------------------

def parse_arguments():
    """Parse command line arguments, dispatching on a subcommand."""
    parser = argparse.ArgumentParser(
        description="Filter/reshape ChEMBL search-results CSVs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    keyword_parser = subparsers.add_parser(
        'keyword-filter',
        help='Keep rows whose full text matches every FILTERS group',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    keyword_parser.add_argument('--input', type=str, required=True, help='Input CSV path (first row is the header)')
    keyword_parser.add_argument(
        '--output', type=str, default=None, help='Output CSV path (default: "<input>_filtered.csv")'
    )

    subtype_parser = subparsers.add_parser(
        'subtype-pairs',
        help='Reshape a per-activity-type reference CSV into clean compound-subtype-value pairs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subtype_parser.add_argument(
        '--input', type=str, required=True,
        help='Input reference CSV path (chembl_id, best_ki_nm_y, Ki_uniprot, best_ic50_nm_y, IC50_uniprot, ...)'
    )
    subtype_parser.add_argument(
        '--output', type=str, default=None, help='Output CSV path (default: "<input>_subtype_pairs.csv")'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    if args.command == 'keyword-filter':
        run_keyword_filter(args)
    elif args.command == 'subtype-pairs':
        run_subtype_pairs(args)


if __name__ == "__main__":
    main()
