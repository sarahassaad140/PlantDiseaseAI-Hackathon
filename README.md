# PlantDiseaseAI — Reliability-Aware Plant Disease Recognition

**Status: hackathon starter; not yet independently clean-clone tested.** This package demonstrates frozen ConvNeXt-Tiny classification with a crop-specific ACCEPT/VERIFY policy and independent YOLO11n tomato disease localization. The user selects tomato or lettuce; the example is tomato. This is a research decision-support prototype, not autonomous diagnosis.

## Use case and problem
In greenhouse/hydroponic monitoring, visual disease symptoms can be difficult to assess consistently. This proof of concept combines image-level classification, selective prediction, and supporting detection overlays to assist human review. Greenhouse environmental context is not implemented in this notebook.

## Setup

**Python 3.10 is recommended.** The notebook was successfully executed in the original `plant_ai` environment. Installation in a completely fresh environment is still being validated.


### 1. Clone the repository

```bash
git clone https://github.com/sarahassaad140/PlantDiseaseAI-Hackathon.git
cd PlantDiseaseAI-Hackathon
```


### 2. Install dependencies

Create and activate a separate Python 3.10 environment, then run:

```bash
python -m pip install -r requirements.txt
```

The requirements reflect the verified local package versions. CPU-only and CUDA-specific PyTorch installations may require platform-specific instructions.

### 3. Download the frozen classifier checkpoint

```bash
python download_checkpoint.py
```

The script retrieves the ConvNeXt-Tiny checkpoint from the `v1.0.0-hackathon` GitHub Release, places it in the expected directory, and verifies its SHA-256 checksum.

The frozen YOLO11n checkpoint is already included in the repository and is also verified by the notebook.

### 4. Execute the notebook

Open `notebooks/01_plant_disease_demo.ipynb` in Jupyter Notebook or JupyterLab.

Select the Python environment containing the installed dependencies, then choose **Restart Kernel and Run All Cells**.

The notebook performs frozen classification, reliability-aware ACCEPT/VERIFY decision-making, and independent YOLO11n detection.

### 5. Expected outputs

The notebook regenerates:

- `results/example_prediction.json`
- `results/example_output.jpg`

The supplied sample image is from a YOLO training split and demonstrates inference functionality, not independent test-set accuracy.

## Data and outputs
`data/sample_input/sample_leaf.jpg` is an example selected from **YOLODetectionCleanV5/train**. This is a training-split example, **not a held-out evaluation**. Dataset origin, exact source license and image redistribution permission must be checked before publishing. `results/example_output.jpg` is the original saved detector output from the local smoke test. The notebook regenerates both JSON and annotated output.

## Evidence and limitations
Frozen YOLO release documentation records test precision 0.893, recall 0.856, mAP@50 0.925 and mAP@50–95 0.836 on 1,516 test images (5,282 instances). These are *reported detector test metrics*, **not** reproducible from the single sample image. The classifier's ACCEPT decision is a threshold-based selective prediction, not a guarantee of correctness. Grad-CAM++ is a qualitative attention aid, not lesion segmentation. The YOLO label set (9 tomato classes) is different from the ConvNeXt label set (10 tomato + 3 lettuce). Neither model identifies crop species automatically. Real-field domain shift remains a limitation.


## Team, licensing, and attribution

**Team attribution:** To be completed with the approved names and roles of all participating team members.

**Dataset provenance and permissions:** The demonstration image originates from the YOLODetectionCleanV5 training split. Its original source, applicable license, and redistribution permission must be confirmed before final submission.

**Repository licensing:** Pending confirmation of third-party code, model, and dataset licensing.


**Frozen classifier checkpoint:** Distributed through the [v1.0.0-hackathon GitHub Release](https://github.com/sarahassaad140/PlantDiseaseAI-Hackathon/releases/tag/v1.0.0-hackathon).


This repository provides a research proof of concept for human decision support. It is not an autonomous or clinically validated plant disease diagnostic system.
 Do not commit private data, unapproved imagery, or the original large experiment archive.
