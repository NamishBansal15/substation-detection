# Comparing Object Detection Models for Electrical Substation Component Mapping

This repository contains the code, processed data, and supporting materials for our research on the automated mapping and large-scale detection of electrical substation components from aerial imagery using computer-vision-based object detection models.

The study trains, benchmarks, and compares **16 object detection models** across multiple architecture families and model sizes. Selected models are subsequently applied to aerial imagery of electrical substations across the United States. The resulting detections are used to characterize the geographic distribution of substation infrastructure at **state** and **FERC regional** scales.

---

## Repository Structure

```text
substation-detection/
├── model-training/
│   ├── yolo_model_training.ipynb
│   ├── rfdetr_model_training.ipynb
│   ├── cascade_resnet50_model_training.ipynb
│   └── cascade_resnet101_model_training.ipynb
│
├── inference-mapping/
│   ├── naip_image_extraction.ipynb
│   ├── model-based-prediction.ipynb
│   └── count-generation.ipynb
│
├── graphs/
│   ├── model_performance_graph.py
│   ├── substation_inference_bar_scatter_graph.py
│   └── substation_inference_choropleth_graphs.py
│
├── results/
│   ├── model_performance.csv
│   ├── model_bootstrapping_CIs.csv
│   ├── state_component_counts.csv
│   └── ferc_region_component_counts.csv
│
├── data/
│   ├── component_predictions.csv
│   ├── image_metadata.csv
│   ├── nerc_gdf.geojson
│   ├── substation_coordinates.parquet
│   └── us_state_populations.csv
│
├── requirements.txt
└── README.md
```

---

## Methodology

The research workflow consists of three primary stages.

### 1. Model Training and Evaluation

Multiple object detection architectures were trained and evaluated for identifying electrical substation components in aerial imagery.

The evaluated model families include:

* **YOLO**
* **Cascade R-CNN**
* **RF-DETR**

Models were trained and evaluated on six annotated classes:

* **Transformer**
* **Circuit Breaker**
* **Reactor**
* **Alternative Energy Systems**
* **Control**
* **Power Lines**

Model performance was evaluated using standard object detection metrics, including:

* **mAP@50** — mean average precision at an intersection-over-union (IoU) threshold of 0.50
* **mAP@50:95** — mean average precision across IoU thresholds from 0.50 to 0.95

The reported aggregate mAP metrics evaluate performance across all six training classes.

Training and evaluation notebooks are available in [`model-training/`](./model-training/).

### 2. Large-Scale Inference and Mapping

Following model evaluation, selected object detection models were applied to aerial imagery of electrical substations across the United States.

**National Agriculture Imagery Program (NAIP)** imagery was used as the primary source of imagery for large-scale inference.

The inference workflow includes:

1. extracting NAIP imagery corresponding to known substation locations;
2. applying trained object detection models to the extracted imagery;
3. generating component-level detections; and
4. aggregating detections geographically.

Control and power-line detections were excluded from the downstream geographic analysis. Accordingly, the large-scale component-mapping analysis focuses on four retained component classes:

* **Transformer**
* **Circuit Breaker**
* **Reactor**
* **Alternative Energy Systems**

Additional information regarding the class-selection methodology and rationale is provided in the associated manuscript.

The relevant notebooks are available in [`inference-mapping/`](./inference-mapping/).

### 3. Geographic and Statistical Analysis

Component detections were aggregated to characterize geographic patterns in electrical substation infrastructure across the United States.

The downstream analysis includes:

* comparisons among models used for large-scale inference;
* component counts across U.S. states;
* component counts across FERC regions; and
* geographic and statistical analyses of detected infrastructure.

Scripts used to generate figures presented in the associated manuscript are available in [`graphs/`](./graphs/).

---

## Dataset

The object detection dataset consists of aerial images containing manually annotated electrical substation components.

The models were trained and evaluated using six component classes:

1. Transformer
2. Circuit Breaker
3. Reactor
4. Alternative Energy Systems
5. Control
6. Power Lines

The subsequent geographic and statistical analyses were restricted to **transformers, circuit breakers, reactors, and alternative energy systems**.

Additional information regarding dataset construction, annotation procedures, class selection, and training/validation/test splitting is provided in the associated manuscript.

The annotated object-detection dataset is not stored directly in this GitHub repository due to its size. Information regarding access to the dataset will be provided with the associated archival release.

NAIP imagery used for large-scale inference is publicly available through the **U.S. Department of Agriculture National Agriculture Imagery Program** and associated geospatial data services.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/NamishBansal15/substation-detection.git
cd substation-detection
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Specific object detection frameworks may have additional installation requirements. See the comments and setup instructions within the corresponding notebooks in [`model-training/`](./model-training/).

---

## Reproducing the Analysis

The general computational workflow is:

```text
Annotated Dataset
       ↓
Model Training
       ↓
Model Evaluation
       ↓
NAIP Image Extraction
       ↓
Large-Scale Model Inference
       ↓
Component Count Generation
       ↓
Geographic / Statistical Analysis
       ↓
Figures and Tables
```

The relevant files should generally be executed in the following order:

1. Run the appropriate notebook in [`model-training/`](./model-training/) to train and evaluate an object detection architecture.
2. Use [`inference-mapping/naip_image_extraction.ipynb`](./inference-mapping/naip_image_extraction.ipynb) to obtain the required NAIP imagery.
3. Run [`inference-mapping/model-based-prediction.ipynb`](./inference-mapping/model-based-prediction.ipynb) to perform component detection.
4. Run [`inference-mapping/count-generation.ipynb`](./inference-mapping/count-generation.ipynb) to aggregate component detections geographically.
5. Use the scripts in [`graphs/`](./graphs/) to reproduce the figures used in the manuscript.

Precomputed numerical results underlying the manuscript are provided in [`results/`](./results/).

---

## Data and Results

The [`data/`](./data/) directory contains processed data used by the inference-mapping and geographic-analysis workflows, including component-level predictions, image metadata, geographic boundaries, substation coordinates, and population data.

The [`results/`](./results/) directory contains machine-readable numerical results underlying the principal analyses presented in the manuscript.

These include:

* model-performance measurements;
* bootstrap confidence intervals;
* state-level component counts; and
* FERC-region component counts.

See the associated manuscript for interpretation and discussion of these results.

---

## Reproducibility and Archival

This repository is intended to provide the code and supporting materials necessary to understand and reproduce the computational analyses reported in the associated research.

A versioned archival snapshot associated with the manuscript will be preserved through **Zenodo**.

**Zenodo DOI:** `[add DOI after creating archival release]`

---

## Associated Publication

### *Comparing Object Detection Models for Electrical Substation Component Mapping*

**Namish Bansal*, Haley Mody*, et al.**

*These authors contributed equally to this work.

**Status:** Manuscript in preparation.

An earlier version of this work is available as an arXiv preprint:

**arXiv:2512.22454**

## Contact

For questions regarding this repository or the associated research, please open an issue through GitHub or contact the authors via namishemail@gmail.com and hdmody09@gmail.com
