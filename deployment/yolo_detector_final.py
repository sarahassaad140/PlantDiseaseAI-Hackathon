from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union
import hashlib

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "yolo_detection"
    / "FINAL_YOLO_RELEASE"
    / "yolo11n_detection_final.pt"
)

EXPECTED_SHA256 = (
    "4c5ea28cd7e175d0e110faedebcce261"
    "3959d415c92543772f1354edff99c4c5"
)

EXPECTED_CLASSES = [
    "Early Blight",
    "Healthy",
    "Late Blight",
    "Leaf Miner",
    "Leaf Mold",
    "Mosaic Virus",
    "Septoria",
    "Spider Mites",
    "Yellow Leaf Curl Virus",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


class FinalYOLODetector:
    """
    Frozen YOLO11n localization subsystem for PlantDiseaseAI.

    Scientific role
    ---------------
    This detector provides object/localization evidence.

    It does NOT:
    - replace the frozen image classifier,
    - override ACCEPT/VERIFY,
    - alter classifier confidence,
    - alter Grad-CAM++,
    - alter greenhouse disease-pressure logic.

    The detector is supporting visual evidence only.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        device: Union[int, str] = 0,
        conf: float = 0.25,
        imgsz: int = 640,
        verify_hash: bool = True,
    ) -> None:

        self.model_path = Path(model_path)
        self.device = device
        self.conf = float(conf)
        self.imgsz = int(imgsz)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO checkpoint not found: {self.model_path}"
            )

        self.sha256 = _sha256(self.model_path)

        if verify_hash and self.sha256 != EXPECTED_SHA256:
            raise RuntimeError(
                "Frozen YOLO checkpoint SHA256 mismatch.\n"
                f"Expected: {EXPECTED_SHA256}\n"
                f"Observed: {self.sha256}"
            )

        self.model = YOLO(str(self.model_path))

        observed_classes = [
            self.model.names[i]
            for i in sorted(self.model.names)
        ]

        if observed_classes != EXPECTED_CLASSES:
            raise RuntimeError(
                "YOLO class-order mismatch.\n"
                f"Expected: {EXPECTED_CLASSES}\n"
                f"Observed: {observed_classes}"
            )

    @staticmethod
    def _prepare_image(
        image: Union[
            str,
            Path,
            Image.Image,
            np.ndarray,
        ]
    ) -> np.ndarray:

        if isinstance(image, (str, Path)):
            path = Path(image)

            if not path.exists():
                raise FileNotFoundError(path)

            array = cv2.imread(str(path))

            if array is None:
                raise ValueError(
                    f"Unable to read image: {path}"
                )

            return array

        if isinstance(image, Image.Image):
            rgb = np.asarray(image.convert("RGB"))
            return cv2.cvtColor(
                rgb,
                cv2.COLOR_RGB2BGR,
            )

        if isinstance(image, np.ndarray):
            if image.ndim != 3:
                raise ValueError(
                    "NumPy image must have shape HxWxC."
                )

            return image.copy()

        raise TypeError(
            "image must be a path, PIL.Image, "
            "or NumPy array."
        )

    def predict(
        self,
        image: Union[
            str,
            Path,
            Image.Image,
            np.ndarray,
        ],
    ) -> Dict[str, Any]:

        source = self._prepare_image(image)

        results = self.model.predict(
            source=source,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )

        if len(results) != 1:
            raise RuntimeError(
                "Expected exactly one YOLO result."
            )

        result = results[0]

        detections: List[Dict[str, Any]] = []

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls.item())
                confidence = float(box.conf.item())

                xyxy = [
                    float(x)
                    for x in box.xyxy[0].tolist()
                ]

                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": self.model.names[
                            class_id
                        ],
                        "confidence": confidence,
                        "bbox_xyxy": xyxy,
                    }
                )

        detections.sort(
            key=lambda x: x["confidence"],
            reverse=True,
        )

        annotated_bgr = result.plot()

        annotated_rgb = cv2.cvtColor(
            annotated_bgr,
            cv2.COLOR_BGR2RGB,
        )

        annotated_pil = Image.fromarray(
            annotated_rgb
        )

        return {
            "detections": detections,
            "count": len(detections),
            "has_detections": bool(detections),
            "annotated_image": annotated_pil,
            "model_sha256": self.sha256,
            "confidence_threshold": self.conf,
            "image_size": self.imgsz,
        }