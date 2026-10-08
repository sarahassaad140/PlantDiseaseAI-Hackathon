# PlantDiseaseAI — Reliability-Aware Plant Disease Recognition

**Status: hackathon starter; not yet independently clean-clone tested.** This package demonstrates frozen ConvNeXt-Tiny classification with a crop-specific ACCEPT/VERIFY policy and independent YOLO11n tomato disease localization. The user selects tomato or lettuce; the example is tomato. This is a research decision-support prototype, not autonomous diagnosis.

## Use case and problem
In greenhouse/hydroponic monitoring, visual disease symptoms can be difficult to assess consistently. This proof of concept combines image-level classification, selective prediction, and supporting detection overlays to assist human review. Greenhouse environmental context is not implemented in this notebook.

## Setup (Python 3.10 recommended)
1. Create a clean environment; install `requirements.txt`. For CPU-only or CUDA-specific PyTorch installation, follow the official PyTorch installation instructions if the default wheels are unsuitable. The pinned list reflects the tested local versions, **not yet a validated fresh install**.
2. Place the original **frozen classifier** checkpoint at `checkpoints/convnext_tiny_targeted_repair_v1/best_balanced_accuracy.pth` (SHA-256 `b4f1ecf003f7710de4454f33babe841da9a302436a74a8a1b19590a68b3ea602`). **Checkpoint hosting/download URL is not yet provided: must be resolved before publishing.**
3. Place the original **frozen YOLO** checkpoint at `outputs/yolo_detection/FINAL_YOLO_RELEASE/yolo11n_detection_final.pt` (SHA-256 `4c5ea28cd7e175d0e110faedebcce2613959d415c92543772f1354edff99c4c5`). The notebook checks both hashes.
4. Start Jupyter from Anaconda `base` and select the `Python (PlantDiseaseAI)` kernel (or your environment with the dependencies). Open `notebooks/01_plant_disease_demo.ipynb`, restart kernel and Run All.

## Data and outputs
`data/sample_input/sample_leaf.jpg` is an example selected from **YOLODetectionCleanV5/train**. This is a training-split example, **not a held-out evaluation**. Dataset origin, exact source license and image redistribution permission must be checked before publishing. `results/example_output.jpg` is the original saved detector output from the local smoke test. The notebook regenerates both JSON and annotated output.

## Evidence and limitations
Frozen YOLO release documentation records test precision 0.893, recall 0.856, mAP@50 0.925 and mAP@50–95 0.836 on 1,516 test images (5,282 instances). These are *reported detector test metrics*, **not** reproducible from the single sample image. The classifier's ACCEPT decision is a threshold-based selective prediction, not a guarantee of correctness. Grad-CAM++ is a qualitative attention aid, not lesion segmentation. The YOLO label set (9 tomato classes) is different from the ConvNeXt label set (10 tomato + 3 lettuce). Neither model identifies crop species automatically. Real-field domain shift remains a limitation.

## Team / licensing / attribution
**TODO before publishing:** add team attribution, approved dataset provenance and license, image redistribution permissions, repository license, presentation reference, and an accessible stable classifier checkpoint download. Do not commit private data, unapproved imagery, or the original large experiment archive.
