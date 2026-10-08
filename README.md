# PlantDiseaseAI — Reliability-Aware Plant Disease Recognition

**Status: Reproducible frozen-inference research prototype — clean-clone and CPU execution verified.**

PlantDiseaseAI demonstrates reliability-aware plant disease recognition using a frozen ConvNeXt-Tiny image classifier, a crop-specific ACCEPT/VERIFY policy, and an independent YOLO11n tomato disease detector.

The repository provides a reproducible inference demonstration for a greenhouse and hydroponic decision-support use case. It does not retrain models, automatically identify crop species, or provide autonomous plant disease diagnosis.

**Reproducibility verification:** On October 8, 2026, the repository was independently cloned and the notebook successfully executed using a newly created Python 3.10 environment on Windows with CPU-only PyTorch. Both model checkpoints passed SHA-256 verification, and the notebook generated its expected outputs without errors.

## 1. Problem and use case

Plant disease symptoms can be difficult to assess consistently from visual observations, particularly when images differ in lighting, background, acquisition conditions, and symptom presentation.

PlantDiseaseAI explores a human-in-the-loop approach that combines:

- **Image-level classification:** ConvNeXt-Tiny predicts a disease or healthy class.
- **Reliability-aware selective prediction:** A frozen, crop-specific policy produces an ACCEPT or VERIFY decision.
- **Supporting localization:** YOLO11n independently predicts disease-associated regions in tomato images.
- **Human review:** Predictions and uncertainty information are intended to assist, not replace, agricultural expertise.

The broader research project also explores explainability and rule-based greenhouse environmental context. The notebook in this repository focuses on frozen visual inference; environmental monitoring and forecasting are not implemented in this demonstration.

## 2. System architecture

### ConvNeXt-Tiny classification

The frozen ConvNeXt-Tiny classifier supports **13 classes**, comprising 10 tomato classes and 3 lettuce classes.

The user must supply the crop identity, either `tomato` or `lettuce`. Crop recognition is not automatic.

The classifier produces a predicted class and calibrated confidence information. Its reliability policy uses crop-specific calibration and uncertainty criteria, including confidence, prediction margin, and normalized entropy, to determine whether a prediction is:

- **ACCEPT:** The prediction meets the frozen reliability criteria.
- **VERIFY:** The prediction should receive additional human review.

An ACCEPT decision does **not** guarantee that a prediction is correct.

### YOLO11n supporting detection

A separate frozen YOLO11n detector supports nine tomato disease/condition categories:

1. Early Blight
2. Healthy
3. Late Blight
4. Leaf Miner
5. Leaf Mold
6. Mosaic Virus
7. Septoria
8. Spider Mites
9. Yellow Leaf Curl Virus

YOLO runs independently on the same input image. Its bounding boxes do not feed into the ConvNeXt classifier and do not override the classifier's ACCEPT/VERIFY decision.

The detector's nine-class label space is different from the classifier's 13-class label space. Its predicted boxes are supporting visual evidence, not verified lesions or independently confirmed disease regions.

## 3. Repository structure

```text
PlantDiseaseAI-Hackathon/
├── README.md
├── requirements.txt
├── download_checkpoint.py
├── .gitignore
├── checkpoints/
│   └── convnext_tiny_targeted_repair_v1/
│       └── best_balanced_accuracy.pth
│          (downloaded separately; not tracked by Git)
├── data/
│   └── sample_input/
│       └── sample_leaf.jpg
├── deployment/
│   ├── __init__.py
│   ├── inference_engine_final_repair_v1.py
│   └── yolo_detector_final.py
├── notebooks/
│   └── 01_plant_disease_demo.ipynb
├── outputs/
│   ├── final_repair_v1_calibration/
│   │   └── reliability_config.json
│   └── yolo_detection/
│       └── FINAL_YOLO_RELEASE/
│           └── yolo11n_detection_final.pt
└── results/
    ├── YOLO_FINAL_METRICS.txt
    ├── example_prediction.json
    └── example_output.jpg
```

The classifier checkpoint is downloaded automatically using the provided script. The smaller YOLO checkpoint is included in the repository.

## 4. Installation and reproducibility

### Prerequisites

- Python 3.10
- Git
- Internet connection for initial dependency and classifier-checkpoint downloads
- Jupyter Notebook or JupyterLab
- Anaconda or Miniconda recommended for environment isolation

An NVIDIA GPU is **not required** for the demonstration. Successful execution has been verified with CPU-only PyTorch on Windows.

### Step 1 — Clone the repository

```bash
git clone https://github.com/sarahassaad140/PlantDiseaseAI-Hackathon.git
cd PlantDiseaseAI-Hackathon
```

### Step 2 — Create an isolated Python environment

Using Anaconda Prompt:

```bash
conda create -n plant_ai_hackathon python=3.10 -y
conda activate plant_ai_hackathon
```

### Step 3 — Install dependencies

```bash
python -m pip install -r requirements.txt
python -m pip check
```

The published dependency versions were successfully installed in a new Python 3.10.22 environment on Windows. The installation passed `pip check` with no broken requirements.

The clean test used PyTorch `2.13.0+cpu` and Torchvision `0.28.0+cpu`. CUDA-enabled environments may require platform-specific PyTorch installation choices.

### Step 4 — Register the Jupyter kernel

```bash
python -m ipykernel install --user --name plant_ai_hackathon --display-name "Python (PlantDiseaseAI Hackathon)"
```

If Jupyter Notebook is not already installed, install it in an appropriate environment:

```bash
python -m pip install notebook
```

Then launch Jupyter from the repository root:

```bash
jupyter notebook
```

### Step 5 — Download the frozen classifier checkpoint

Run from the repository root:

```bash
python download_checkpoint.py
```

The script downloads the frozen ConvNeXt-Tiny checkpoint from the [v1.0.0-hackathon GitHub Release](https://github.com/sarahassaad140/PlantDiseaseAI-Hackathon/releases/tag/v1.0.0-hackathon), places it in the expected directory, and verifies its SHA-256 checksum.

If a valid checkpoint already exists, the script recognizes it and avoids downloading it again.

**Frozen classifier SHA-256:**

```text
b4f1ecf003f7710de4454f33babe841da9a302436a74a8a1b19590a68b3ea602
```

**Frozen YOLO11n SHA-256:**

```text
4c5ea28cd7e175d0e110faedebcce2613959d415c92543772f1354edff99c4c5
```

Both checkpoint hashes are checked by the notebook before inference. A mismatched checkpoint must not be silently substituted.

### Step 6 — Execute the notebook

In Jupyter, open:

`notebooks/01_plant_disease_demo.ipynb`

Select the kernel **Python (PlantDiseaseAI Hackathon)**, then choose **Restart Kernel and Run All Cells**.

The notebook:

1. Loads and displays the example tomato image.
2. Verifies the frozen classifier and YOLO checkpoint hashes.
3. Runs ConvNeXt-Tiny classification and the reliability gate.
4. Runs independent YOLO11n localization.
5. Displays the annotated image.
6. Saves reproducible example outputs.

The notebook supports execution when launched from the repository root or its `notebooks/` directory.

## 5. Demonstration input and expected outputs

### Sample input

`data/sample_input/sample_leaf.jpg`

The included image comes from a **YOLO training split**. It is used to demonstrate the inference pipeline and must not be interpreted as a held-out evaluation example.

The demonstration supplies `tomato` as the crop identity.

### Example classification output

```text
Class: Tomato yellow leaf curl virus
Confidence: approximately 0.794
Reliability decision: ACCEPT
```

### Example detection output

```text
YOLO detections: 18
Predicted category: Yellow Leaf Curl Virus
```

The YOLO detector may produce overlapping boxes and labels. These are retained as model predictions rather than treated as independently verified lesions.

The results are illustrative. Small numerical or visualization differences may occur across hardware and software environments.

### Generated files

The notebook regenerates:

- `results/example_prediction.json` — structured inference results and metadata.
- `results/example_output.jpg` — annotated YOLO detection output.

Example output files are included so that users can inspect the expected artifact formats before running the notebook.

A successful single-image demonstration does **not** reproduce the research evaluation metrics.

## 6. Dataset provenance, licensing, and attribution

### Demonstration image source

The sample image was selected from the `YOLODetectionCleanV5` training split. The original detection dataset is documented as:

**Tomato Leaf Disease — Roboflow Universe, version 63**

- **Workspace:** `bryan-b56jm`
- **Project:** `tomato-leaf-disease-ssoha`
- **Version:** 63
- **License recorded in source `data.yaml`:** CC BY 4.0

Original dataset:

https://universe.roboflow.com/bryan-b56jm/tomato-leaf-disease-ssoha/dataset/63

A corresponding dataset is also available on Kaggle:

https://www.kaggle.com/datasets/kpoviesistphane/tomato-leaf-disease-detection

The Kaggle listing displays the license as **Unknown**, whereas the dataset's `data.yaml` records **CC BY 4.0** for the Roboflow source. The Roboflow metadata provides the more specific source attribution, but any additional third-party image rights or restrictions must still be respected.

### Attribution

The demonstration image is attributed to the **Tomato Leaf Disease** dataset, Roboflow Universe project `tomato-leaf-disease-ssoha`, workspace `bryan-b56jm`, version 63, with a CC BY 4.0 license recorded in its dataset metadata.

The image is redistributed here as an inference example. The repository also contains a derivative visualization with predicted YOLO bounding boxes and labels.

The annotated output is a modification of the source image produced by the frozen YOLO model.

CC BY 4.0 license information:

https://creativecommons.org/licenses/by/4.0/

### Other research data

The broader ConvNeXt-Tiny classifier was developed using multiple agricultural image domains. The complete training datasets and experiment archive are **not** redistributed in this repository.

The demonstration sample must not be used as evidence of classifier or detector generalization.

## 7. Research evidence and limitations

### Frozen YOLO11n detection evaluation

The frozen YOLO release documentation reports the following metrics on its reserved test set:

| Metric | Reported value |
|---|---:|
| Precision | 0.893 |
| Recall | 0.856 |
| mAP@50 | 0.925 |
| mAP@50–95 | 0.836 |
| Test images | 1,516 |
| Test instances | 5,282 |

The complete release metrics are recorded in `results/YOLO_FINAL_METRICS.txt`.

These are previously obtained frozen detector test results. They are **not** reproduced by the single-image notebook demonstration.

The detector's validation metrics were lower than its final test metrics; performance differences across evaluation splits should be interpreted cautiously.

### Classifier evaluation context

Historical evaluation of the broader PlantDiseaseAI research system includes:

| Evaluation | Reported accuracy |
|---|---:|
| Internal multi-domain test | 98.09% |
| Supplementary tomato external audit | 71.20% |
| Held-out lettuce evaluation | 91.15% |
| Separate 70-image field-style stress test | 31.43% |

These results correspond to different datasets and evaluation conditions. They must not be treated as interchangeable measures of real-world performance.

In particular, the high internal test accuracy should **not** be presented as expected field accuracy. The supplementary tomato external audit was influenced by earlier domain adaptation and is not a perfectly untouched final evaluation.

The field-style stress test highlights the continuing difficulty of agricultural domain shift.

### Reliability-aware prediction

The classifier applies frozen crop-specific calibration and selective-prediction criteria.

An ACCEPT result indicates that the prediction satisfies the defined reliability thresholds. It is **not** a guarantee of diagnostic correctness.

VERIFY indicates that additional review is appropriate. Thresholds are not tuned using the final test data in this demonstration.

### Explainability

The broader system supports Grad-CAM++ as a qualitative attention aid. Attention maps are not ground-truth lesion masks and should not be interpreted as proof that a diagnosis is correct.

Grad-CAM++ visualization is not required to reproduce the notebook's principal inference demonstration.

### Scope limitations

- The demonstration does not retrain or fine-tune either model.
- The demonstration does not independently recalculate reported test metrics.
- The user must manually provide crop identity.
- YOLO supports tomato categories only.
- ConvNeXt and YOLO operate independently.
- The detector's bounding boxes are predictions, not verified disease regions.
- Real-world lighting, background, crop variety, acquisition conditions, and domain shift may substantially affect performance.
- The greenhouse environmental-context subsystem is not executed in this notebook.
- The prototype is not an autonomous disease diagnosis or treatment recommendation system.

## 8. Reproducibility verification

The following checks were completed on October 8, 2026:

| Check | Result |
|---|---|
| Public GitHub clone | Passed |
| Automatic classifier checkpoint download | Passed |
| Classifier SHA-256 verification | Passed |
| YOLO SHA-256 verification | Passed |
| Fresh-clone notebook execution | Passed |
| New Python 3.10 environment creation | Passed |
| Clean dependency installation | Passed |
| Dependency integrity check (`pip check`) | Passed |
| CPU-only PyTorch inference | Passed |
| ConvNeXt classification and reliability gate | Passed |
| YOLO11n inference and annotation | Passed |
| JSON and annotated image generation | Passed |

These checks establish that the demonstration executed successfully in the tested Windows CPU environment. They do not establish universal compatibility across all operating systems or hardware configurations.

## 9. Team and contributions

**Project:** PlantDiseaseAI — Reliability-Aware Plant Disease Recognition

**Team members and roles:**

- **Sarah Assaad — Lead AI Researcher & Machine Learning Architect** —  Led the end-to-end AI development, including multi-domain deep learning, model optimization, robustness evaluation, reliability-aware prediction, explainable AI, YOLO integration, and reproducible deployment
- **Pr.Mohamad Khalil  — Professor, Lebanese University; Director, AZM Research Center for Biotechnology** — Academic and biotechnology expertise, affiliated with the École Doctorale des Sciences et Technologies and Université de Technologie de Compiègne.
- **Dr. Fatima Yahya  — PhD, Senior Researcher in Chemistry and Environmental Science, Lebanese University** — Scientific expertise in chemistry, environmental research, and greenhouse-related applications.

## 10. Code, model, and data licensing

The demonstration image's source metadata identifies the Roboflow dataset license as **CC BY 4.0**, as described in Section 6.

This attribution does not automatically establish that all repository source code, pretrained model components, trained checkpoint weights, or other research datasets share the same license.

Repository-wide licensing and any third-party notices should be reviewed before adding a general software license.

No license for the complete repository is asserted solely on the basis of the demonstration image's CC BY 4.0 metadata.

## 11. Frozen checkpoint release

The frozen ConvNeXt-Tiny classifier checkpoint is distributed separately to keep the Git repository lightweight.

[Download or inspect the frozen classifier release — v1.0.0-hackathon](https://github.com/sarahassaad140/PlantDiseaseAI-Hackathon/releases/tag/v1.0.0-hackathon)

The `download_checkpoint.py` script automates retrieval and integrity verification.

## 12. Responsible use

PlantDiseaseAI is a research proof of concept intended to support human agricultural decision-making.

Its predictions, confidence scores, reliability decisions, and detection overlays must be interpreted in context. The system is not a substitute for expert diagnosis, laboratory testing, or appropriate crop-management procedures.

Do not commit private data, unapproved imagery, credentials, or the original large experiment archive to this repository.