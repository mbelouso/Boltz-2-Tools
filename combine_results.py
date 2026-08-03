'''
Combines per-receptor collate_boltz2.py output CSVs into a single wide table, joined on
chembl_id, with columns prefixed by receptor name (e.g. M1_affinity_pred_value,
M2_affinity_pred_value, ...).

Usage:
python combine_results.py --parent-dir /Users/ydon0043/Muscarinic_boltz_project --output combined_results.csv

Expects a directory layout like:
    <parent-dir>/M1_MuscarinicSet/boltz_results.csv
    <parent-dir>/M2_MuscarinicSet/boltz_results.csv
    ...
Any subfolder of <parent-dir> whose name starts with M<number> (e.g. M1_MuscarinicSet,
M2_..., M10_...) is treated as a receptor and picked up automatically -- no need to list
folders explicitly. Compounds are kept even if they're missing from some receptors (outer
join); missing values show up as blank/NaN in the combined CSV.

Optionally pass --input to left-join a subtype-pairs reference table (output of
filter_searched_results.py's subtype-pairs subcommand: chembl_id, M1_Ki_nm, M1_IC50_nm,
M1_EC50_nm, ..., M5_EC50_nm) onto the combined results, keyed on chembl_id. Only compounds
already in the combined results are kept. When given, a second file is written alongside
--output: "<output>_with_reference.csv" (or the path given via --reference-output) contains
the combined results plus the reference columns, while --output itself always stays the
plain per-receptor combined table.
'''

import os
import re
import argparse
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RECEPTOR_PATTERN = re.compile(r'^M\d+')


def parse_arguments():
    """Parse command line arguments for configuration options."""
    parser = argparse.ArgumentParser(
        description="Combine per-receptor collate_boltz2.py results into one wide table, joined on chembl_id",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--parent-dir',
        type=str,
        default='.',
        help='Parent directory containing the receptor subfolders (e.g. M1_MuscarinicSet, M2_MuscarinicSet, ...)'
    )

    parser.add_argument(
        '--results-filename',
        type=str,
        default='boltz_results.csv',
        help='Filename to look for inside each receptor subfolder'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='combined_results.csv',
        help='Output CSV path'
    )

    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help=(
            'Optional subtype-pairs reference CSV (output of filter_searched_results.py\'s '
            'subtype-pairs subcommand: chembl_id, M1_Ki_nm, M1_IC50_nm, ..., M5_EC50_nm) to '
            'left-join onto the combined results, keyed on chembl_id. When given, a second '
            'output file (see --reference-output) is written with this data attached, in '
            'addition to the plain combined --output file.'
        )
    )

    parser.add_argument(
        '--reference-output',
        type=str,
        default=None,
        help=(
            'Output CSV path for the combined results with reference data attached. Only '
            'used when --input is given. Defaults to "<output>_with_reference.csv".'
        )
    )

    return parser.parse_args()


def find_receptor_folders(parent_dir: Path) -> dict:
    """Find subfolders of parent_dir whose name starts with M<number>, mapped to receptor prefix -> path."""
    receptor_folders = {}
    for entry in sorted(os.listdir(parent_dir)):
        entry_path = parent_dir / entry
        if not entry_path.is_dir():
            continue
        match = RECEPTOR_PATTERN.match(entry)
        if match:
            receptor_folders[match.group(0)] = entry_path
        else:
            logger.debug(f"Skipping folder (doesn't match ^M<number>): {entry}")
    return receptor_folders


def load_receptor_results(receptor: str, folder: Path, results_filename: str) -> pd.DataFrame:
    """Load one receptor's results CSV, renaming all non-chembl_id columns with the receptor prefix."""
    csv_path = folder / results_filename
    if not csv_path.exists():
        logger.warning(f"No '{results_filename}' found in {folder}, skipping receptor {receptor}")
        return None

    df = pd.read_csv(csv_path)

    if 'chembl_id' not in df.columns:
        logger.error(f"'{csv_path}' has no chembl_id column, skipping receptor {receptor}")
        return None

    duplicate_count = df['chembl_id'].duplicated().sum()
    if duplicate_count > 0:
        logger.warning(
            f"{receptor}: {duplicate_count} duplicate chembl_id rows in '{csv_path}' "
            f"(likely multiple models per compound) -- keeping the first row per chembl_id"
        )
        df = df.drop_duplicates(subset='chembl_id', keep='first')

    rename_map = {col: f"{receptor}_{col}" for col in df.columns if col != 'chembl_id'}
    df = df.rename(columns=rename_map)

    logger.info(f"{receptor}: loaded {len(df)} compounds from {csv_path}")
    return df


def combine_receptor_results(parent_dir: str, results_filename: str) -> pd.DataFrame:
    """Find all receptor folders under parent_dir, load and outer-merge their results on chembl_id."""
    parent_path = Path(parent_dir)
    receptor_folders = find_receptor_folders(parent_path)

    if not receptor_folders:
        raise ValueError(f"No receptor folders (matching ^M<number>) found under {parent_dir}")

    ordered_receptors = sorted(receptor_folders.keys(), key=lambda r: int(re.search(r'\d+', r).group()))
    logger.info(f"Found {len(ordered_receptors)} receptor folders: {ordered_receptors}")

    combined_df = None
    receptors_used = []

    for receptor in ordered_receptors:
        receptor_df = load_receptor_results(receptor, receptor_folders[receptor], results_filename)
        if receptor_df is None:
            continue

        receptors_used.append(receptor)
        if combined_df is None:
            combined_df = receptor_df
        else:
            combined_df = pd.merge(combined_df, receptor_df, on='chembl_id', how='outer')

    if combined_df is None:
        raise ValueError("No receptor results were successfully loaded")

    logger.info(f"Combined {len(combined_df)} compounds across {len(receptors_used)} receptors: {receptors_used}")
    return combined_df


def load_reference_data(reference_csv: str) -> pd.DataFrame:
    """Load a subtype-pairs reference table (output of filter_searched_results.py's
    subtype-pairs subcommand: chembl_id, M1_Ki_nm, M1_IC50_nm, M1_EC50_nm, ..., M5_EC50_nm)."""
    df = pd.read_csv(reference_csv)

    if 'chembl_id' not in df.columns:
        raise ValueError(f"'{reference_csv}' has no chembl_id column")

    duplicate_count = df['chembl_id'].duplicated().sum()
    if duplicate_count > 0:
        logger.warning(
            f"Reference CSV: {duplicate_count} duplicate chembl_id rows in '{reference_csv}' "
            f"-- keeping the first row per chembl_id"
        )
        df = df.drop_duplicates(subset='chembl_id', keep='first')

    logger.info(f"Loaded reference data for {len(df)} compounds from '{reference_csv}'")
    return df


def attach_reference_data(combined_df: pd.DataFrame, reference_csv: str) -> pd.DataFrame:
    """Left-merge the subtype-pairs reference data onto combined_df, keyed on chembl_id.
    Compounds only present in the reference CSV (never docked) are dropped; docked
    compounds with no reference match get NaN in the new columns."""
    reference_df = load_reference_data(reference_csv)
    before = len(combined_df)
    merged = pd.merge(combined_df, reference_df, on='chembl_id', how='left')

    new_columns = [col for col in reference_df.columns if col != 'chembl_id']
    matched = merged[new_columns].notna().any(axis=1).sum() if new_columns else 0
    logger.info(f"Attached reference data: {matched}/{before} compounds matched in '{reference_csv}'")

    return merged


def main():
    """Main execution function."""
    args = parse_arguments()
    combined_df = combine_receptor_results(args.parent_dir, args.results_filename)

    combined_df.to_csv(args.output, index=False)
    logger.info(f"Combined results exported to '{args.output}'")

    if args.input:
        reference_output = args.reference_output
        if reference_output is None:
            output_path = Path(args.output)
            reference_output = str(output_path.with_name(f"{output_path.stem}_with_reference{output_path.suffix}"))

        with_reference_df = attach_reference_data(combined_df, args.input)
        with_reference_df.to_csv(reference_output, index=False)
        logger.info(f"Combined results with reference data exported to '{reference_output}'")


if __name__ == "__main__":
    main()
