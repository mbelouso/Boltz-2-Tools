# Boltz-2-Tools
Tools for Boltz-2 CADD work.

## Anaconda environment setup

From the repository root (`Boltz-2-Tools`), create and activate a dedicated environment:

```bash
conda create -n boltz2 python=3.10 -y
conda activate boltz2
```

Install core scientific packages with conda:

```bash
conda install -y -c conda-forge \
	pandas pyyaml numpy matplotlib seaborn biopython modelcif rdkit scikit-learn ipython joblib
```

Install the ChEMBL client with pip:

```bash
pip install chembl-webresource-client
```

Optional quick check:

```bash
python -c "import pandas, yaml, numpy, matplotlib, seaborn, Bio, modelcif, rdkit, sklearn, joblib; print('Environment OK')"
```

Notes:
- `sqlite3` is part of the Python standard library and does not need a separate install.
- If you prefer reproducibility, export your environment after setup:

```bash
conda env export --no-builds > environment.yml
```

## Using `Filter_Boltz2.py`

`Filter_Boltz2.py` parses Boltz-2 prediction outputs, computes hydrogen bonds and distance metrics, merges those with confidence/affinity JSON data, filters models by thresholds, aligns filtered structures, and writes summary files and plots.

### Expected working directory and input layout

Run the script from the repository root where your Boltz results and YAML files are available.

Expected inputs include:
- Folders matching `boltz_results_*`
- Inside each: `predictions/<id>/` with `confidence_*.json`, `affinity_*.json`, and `*_model_*.pdb`
- YAML directories (`yaml1` ... `yaml8` or `yaml`) for SMILES extraction

### Basic run

```bash
python Filter_Boltz2.py
```

This uses defaults:
- `confidence-value`: `0.85`
- `affinity-value`: `-1.5`
- `probability-binary-value`: `0.6`
- `binding-site-residue1`: `157`
- `binding-site-residue2`: `None`
- `binding-site-residue3`: `None`
- `distance-threshold`: `15.0`

### Example with explicit thresholds

```bash
python Filter_Boltz2.py \
	--confidence-value 0.90 \
	--affinity-value -2.0 \
	--probability-binary-value 0.7 \
	--binding-site-residue1 120 \
	--distance-threshold 12.0 \
	--plot-residue1 120 --plot-residue2 157
```

### Allow missing affinity metrics

If some models do not include `affinity_pred_value` or `affinity_probability_binary`, use:

```bash
python Filter_Boltz2.py --allow-empty-affinity
```

With this flag, filtering ignores affinity thresholds and uses confidence, hydrogen bonds, and residue distance criteria.

### Run from config file

You can provide a JSON config file. Keys should use underscore names (matching the script config object), for example:

```json
{
	"confidence_value": 0.9,
	"affinity_value": -2.0,
	"probability_binary_value": 0.7,
	"binding_site_residue1": 120,
	"binding_site_residue2": 150,
	"binding_site_residue3": null,
	"distance_threshold": 12.0,
	"plot_residue1": 120,
	"plot_residue2": 157,
	"allow_empty_affinity": false
}
```

Run with:

```bash
python Filter_Boltz2.py --config-file config.json
```

### Main outputs

The pipeline generates (depending on data availability):
- `boltz_results_combined.csv`
- `hydrogen_bond_data.csv`
- `distance_matrices.csv`
- `reduced_distance_data.csv`
- `boltz_results_combined_with_hbonds_distances.csv`
- `Final_Filtered_models.csv`
- `data_summary.json`
- `boltz_results_distribution.png`
- `distance_distribution_plots.png`
- `hbonds_vs_affinity.png`
- Archive ZIP: `boltz_results_archive_confidence<...>_affinity<...>.zip`
- Output folders: `combined_models/`, `distance_filtered/`, `filtered_models/`, `filtered_aligned/`

### Troubleshooting

- If no models are filtered, try relaxing thresholds (`--confidence-value`, `--affinity-value`, `--distance-threshold`).
- If distance plots are empty, ensure selected `--plot-residue*` values exist in `distance_matrices.csv` columns.
- If affinity columns are missing in your Boltz output, use `--allow-empty-affinity`.


