# Comparing Object Detection Models for Electrical Substation Component Mapping

This repository contains the reproducible code, processed data, and supporting materials for research on automated mapping of electrical substation components from aerial imagery using object-detection models.

The study benchmarks **16 object-detection models** spanning YOLO, Cascade R-CNN, and RF-DETR architectures. Selected models are then applied to U.S. National Agriculture Imagery Program (NAIP) imagery, and the resulting detections are analyzed at state and FERC-region scales.

## Repository structure

```text
substation-detection/
├── pyproject.toml
├── src/
│   └── substation_detection/
│       ├── __init__.py
│       ├── cli.py
│       ├── paths.py
│       └── validation.py
├── tests/
│   └── test_repository.py
├── scripts/
│   └── reproduce_results.py
├── model-training/
│   ├── yolo-model-training.ipynb
│   ├── rfdetr_model_training.ipynb
│   ├── cascade_resnet50_model_training.ipynb
│   └── cascade_resnet101_model_training.ipynb
├── inference-mapping/
│   ├── naip-image-extraction.ipynb
│   ├── model-based-prediction.ipynb
│   └── count-generation.ipynb
├── graphs/
│   ├── model_performance_graph.py
│   ├── substation_inference_bar_scatter_graph.py
│   └── substation_inference_choropleth_graphs.py
├── data/
└── results/
```

## Quick start with UV

Python 3.10 or newer is required. UV is the recommended environment and package manager.

```bash
git clone https://github.com/NamishBansal15/substation-detection.git
cd substation-detection

uv lock
uv sync --group dev
uv run reproduce-substation-analysis check
uv run pytest
```

`uv lock` creates `uv.lock`. Commit that lockfile to the repository after generating it so future users resolve the same environment.

To regenerate the manuscript figures from the included processed data:

```bash
uv run reproduce-substation-analysis figures
```

To validate the repository and then regenerate the figures:

```bash
uv run reproduce-substation-analysis all
```

The equivalent convenience script is:

```bash
uv run python scripts/reproduce_results.py
```

Generated figures are written to `results/`.

## Reproducibility levels

The repository supports two related workflows.

### Reproducing the downstream analysis

This is the primary local reproducibility path. The processed inputs needed for geographic/statistical analysis are versioned in `data/`, and the numerical outputs underlying the manuscript are versioned in `results/`.

The package command

```bash
uv run reproduce-substation-analysis check
```

checks that required files are present, prediction and metadata identifiers agree, and FERC totals are internally consistent. The test suite provides additional automated checks.

### Re-running model training and large-scale inference

The training and inference notebooks document the original experimental workflows. They depend on external resources that are intentionally not stored in this repository, including the annotated training dataset, model weights, NAIP imagery, GPU runtimes, and service credentials.

Install only the model-specific extras you need:

```bash
# YOLO notebooks
uv sync --extra notebooks --extra yolo

# RF-DETR notebooks
uv sync --extra notebooks --extra rfdetr

# Cascade R-CNN notebooks
uv sync --extra notebooks --extra cascade
```

### Cascade R-CNN / Detectron2

Cascade R-CNN experiments use Detectron2. Detectron2 is intentionally not
included in the project's universal UV environment because it must be built
against a compatible PyTorch, compiler, and, where applicable, CUDA
configuration.

The `cascade` extra installs the remaining Python dependencies used by the
Cascade R-CNN notebooks:

```bash
uv sync --extra notebooks --extra cascade

## Methodology

### 1. Model training and evaluation

The evaluated model families are:

- YOLO
- Cascade R-CNN
- RF-DETR

Models were trained and evaluated on six annotated classes:

- Transformer
- Circuit Breaker
- Reactor
- Alternative Energy Systems
- Control
- Power Lines

Performance was evaluated using mAP@50 and mAP@50:95. Aggregate model metrics therefore reflect all six training classes. Training/evaluation notebooks are in `model-training/`.

### 2. Large-scale inference and mapping

Selected models were applied to NAIP aerial imagery associated with U.S. electrical substations. The workflow consists of NAIP image extraction, model inference, component-level prediction generation, and geographic aggregation.

Control and power-line detections were excluded from the downstream geographic analysis. The mapping analysis therefore retains four classes:

- Transformer
- Circuit Breaker
- Reactor
- Alternative Energy Systems

The corresponding notebooks are in `inference-mapping/`.

### 3. Geographic and statistical analysis

Detections are aggregated to characterize geographic patterns in substation infrastructure at state and FERC-region scales. Figure-generation scripts are in `graphs/`. They use repository-relative paths so they can be run from a cloned checkout without editing local file locations.

## Data

The `data/` directory contains processed inputs used by the reproducible downstream analysis:

- `component_predictions.csv` — component-level prediction counts by image
- `image_metadata.csv` — image identifiers and geographic metadata
- `nerc_gdf.geojson` — regional boundary geometry used by the geographic analysis
- `substation_coordinates.parquet` — substation coordinate data
- `us_state_populations.csv` — population data used in normalized analyses

The full annotated object-detection training dataset and national NAIP imagery are not stored in GitHub because of their size. Dataset-access information should accompany the archival release/manuscript where applicable.

## Results

The `results/` directory contains machine-readable outputs underlying the principal manuscript analyses:

- `model_performance.csv`
- `model_bootstrapping_CIs.csv`
- `state_component_counts.csv`
- `ferc_region_component_counts.csv`

Figures generated by the reproduction command are also written to this directory.

## Tests

Run:

```bash
uv run pytest
```

The tests check core repository invariants, including agreement between image metadata and prediction identifiers and consistency of FERC totals with the retained component classes.

## Reproducibility and archival

This repository is structured as an installable Python project and is intended to be paired with a committed `uv.lock` file for a frozen environment. A versioned archival snapshot associated with the manuscript will be preserved through Zenodo.

**Zenodo DOI:** add after the archival release is created.

## Associated publication

### *Comparing Object Detection Models for Electrical Substation Component Mapping*

**Namish Bansal\*, Haley Mody\*, et al.**

\*These authors contributed equally to this work.

**Status:** Manuscript in preparation.

An earlier version of this work is available as arXiv:2512.22454.

## Contact

For questions about the repository or associated research, please open an issue on GitHub or contact the authors using the contact information provided with the associated publication.
