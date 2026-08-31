# Comparing Object Detection Models for Electrical Substation Component Mapping 

This repository is a supplement to our research on the automated mapping and large-scale detection of electrical substation components from Google Earth-based aerial imagery, 
utilizing computer-vision-based object detection models. 

This study trains, benchmarks, and compares 16 object detection models from varying sizes and families. These models are applied to substation inference mapping across the United States, 
and the resulting detections are utilized to characterize the geographic distribution of substation infrastructure at state and FERC-based regional scales.

# Repository Structure
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
│   ├── state_component_counts.csv
│   ├── region_component_counts.csv
│   └── statistical_analysis.csv
│
├── data/
│   ├── substation_coordinates.csv
│   └── per_class_imagery.zip
│
├── requirements.txt
├── LICENSE
└── README.md
```

# Methodology
The research workflow consists of three primary stages:

**1. Model Training and Evaluation**

Multiple object detection architectures were trained and evaluated for identifying electrical substation components in aerial imagery.

The evaluated model families include:

- YOLO
- Cascade R-CNN
- RF-DETR

The models were evaluated using standard object detection metrics, including mean average precision at an intersection-over-union threshold of 0.50 (mAP@50) and mean average precision across IoU thresholds from 0.50 to 0.95 (mAP@50:95).

Training and evaluation notebooks are available in **model-training/**.

**2. Large-Scale Inference and Mapping**

Following model evaluation, selected object detection models were applied to aerial imagery of electrical substations across the United States.

National Agriculture Imagery Program (NAIP) imagery was used as the primary source of aerial imagery.

The inference workflow includes:

- extracting the corresponding NAIP imagery from a CSV file of substation locations;
- applying trained object detection models to extracted imagery;
- generating component-level detections and aggregating detections geographically.

Model detections were aggregated to examine geographic patterns in electrical substation infrastructure, with analysis including comparison across US states and FERC regions.
The relevant notebooks are available in **inference-mapping/**.

**3. Geographic and Statistical Analysis**

Model detections were aggregated to examine geographic patterns in electrical substation infrastructure. Analysis includes comparisons across models utilized, counts of US states and FERC regions. 

Scripts used to generate the figures presented in the manuscript are available in **graphs/**.

## Dataset

The object detection dataset consists of aerial images containing annotated electrical substation components, and is located in **dataset/**.

The substation component classes detected are transformer, circuit breaker, reactor, and alternative energy systems.

Additional information regarding dataset construction, annotation procedures, and training/validation/test splitting is provided in the associated manuscript.

NAIP imagery is available through the U.S. Department of Agriculture and associated public geospatial data services.

## Installation

Clone this repository:

git clone https://github.com/NamishBansal15/substation-detection.git
cd substation-detection

Install the required Python packages:

pip install -r requirements.txt

Specific object detection frameworks may have additional installation requirements. See the comments and setup instructions within the corresponding model-training notebooks.

## Reproducing the Analysis

The general workflow for reproducing the study is:

Dataset
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

The relevant files should generally be executed in the following order:

Run the appropriate notebook in model-training/ to train and evaluate an object detection architecture using the data from the ZIP file in dataset/.
Use inference-mapping/naip_image_extraction.ipynb to obtain the required aerial imagery.
Run inference-mapping/model-based-prediction.ipynb to perform component detection.
Run inference-mapping/count-generation.ipynb to aggregate the resulting detections.
Use the scripts in graphs/ to reproduce figures used in the manuscript.

Precomputed numerical results used in the manuscript are provided in results/ where redistribution is appropriate.

## Results

Machine-readable results underlying the principal analyses are provided in the results/ directory.

These files include model-performance measurements and aggregated geographic results used to generate figures and tables in the associated manuscript.

See the manuscript for interpretation and discussion of these results.

## Reproducibility

This repository is intended to provide the code and supporting materials necessary to understand and reproduce the computational analyses reported in the associated research.

A versioned archival snapshot of the repository associated with the manuscript is preserved through Zenodo.

Zenodo DOI: [add DOI after creating the Zenodo release]

## Associated Publication

**[Comparing Object Detection Models for Electrical Substation Component Mapping]**

Namish Bansal*, Haley Mody*, et al.
*These authors contributed equally to this work.*

**Status:** Manuscript in preparation.

An earlier version of this work is available as an arXiv preprint: [[arXiv link]](https://arxiv.org/abs/2512.22454).

## Citation

If you use this code or the associated research, please cite the corresponding paper:

Bansal, N., Haley M., et al. ([YEAR]).
"[PAPER TITLE]."
[JOURNAL].
[DOI]

A machine-readable citation is also provided in CITATION.cff.

## License

See the LICENSE file for information regarding permitted use and redistribution of the code contained in this repository.

Licensing or usage conditions applicable to third-party datasets, imagery, pretrained models, or software remain governed by their respective providers.

Contact

For questions regarding this repository or the associated research, please open an issue through GitHub or contact the authors through the information provided in the associated publication.
