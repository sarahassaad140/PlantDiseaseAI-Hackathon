# -*- coding: utf-8 -*-

"""
==============================================================================
Plant Disease AI
FINAL FROZEN MULTICROP INFERENCE ENGINE — REPAIR V1
==============================================================================

Final checkpoint
----------------
checkpoints/convnext_tiny_targeted_repair_v1/
    best_balanced_accuracy.pth

Frozen reliability configuration
--------------------------------
outputs/final_repair_v1_calibration/
    reliability_config.json

Supported crops
---------------
- tomato : 10 classes
- lettuce: 3 classes

Important deployment behavior
-----------------------------
1. Crop identity is supplied by the app/user.
   This classifier does NOT automatically identify tomato vs lettuce.

2. All uploaded images are internally normalized to the model input:
       Resize(236) -> CenterCrop(224) -> ImageNet normalization

   The original upload dimensions do NOT need to be 224x224.

3. Tomato and lettuce use separate:
       - temperature
       - confidence gate
       - margin gate
       - entropy gate

4. If the prediction fails the reliability policy:
       accepted = False
   The engine still returns the leading class, but deployment should present it
   as an unverified prediction rather than an accepted diagnosis.

5. Grad-CAM++ is provided as a QUALITATIVE ATTENTION AID ONLY.
   It must not be presented as accurate lesion segmentation/localization.

No training.
No calibration fitting.
No threshold search.

==============================================================================
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from PIL import Image, ImageOps

from torchvision import transforms
from torchvision.models import convnext_tiny


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "convnext_tiny_targeted_repair_v1"
    / "best_balanced_accuracy.pth"
)

RELIABILITY_CONFIG_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "final_repair_v1_calibration"
    / "reliability_config.json"
)


# =============================================================================
# CLASS MAPPING
# =============================================================================

CHECKPOINT_CLASSES = [
    "Lettuce___Bacterial_leaf_spot",
    "Lettuce___Fungal_disease",
    "Lettuce___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

LETTUCE_CLASSES = CHECKPOINT_CLASSES[0:3]
TOMATO_CLASSES = CHECKPOINT_CLASSES[3:13]

LETTUCE_CHECKPOINT_INDICES = [0, 1, 2]
TOMATO_CHECKPOINT_INDICES = list(range(3, 13))

CROP_CLASS_NAMES = {
    "tomato":
        TOMATO_CLASSES,

    "lettuce":
        LETTUCE_CLASSES,
}

CROP_CHECKPOINT_INDICES = {
    "tomato":
        TOMATO_CHECKPOINT_INDICES,

    "lettuce":
        LETTUCE_CHECKPOINT_INDICES,
}


# =============================================================================
# USER-FACING DISPLAY NAMES
# =============================================================================

DISPLAY_NAMES = {
    "Lettuce___Bacterial_leaf_spot":
        "Lettuce bacterial leaf spot",

    "Lettuce___Fungal_disease":
        "Lettuce fungal disease",

    "Lettuce___healthy":
        "Healthy lettuce",

    "Tomato___Bacterial_spot":
        "Tomato bacterial spot",

    "Tomato___Early_blight":
        "Tomato early blight",

    "Tomato___Late_blight":
        "Tomato late blight",

    "Tomato___Leaf_Mold":
        "Tomato leaf mold",

    "Tomato___Septoria_leaf_spot":
        "Tomato Septoria leaf spot",

    "Tomato___Spider_mites Two-spotted_spider_mite":
        "Two-spotted spider mite damage",

    "Tomato___Target_Spot":
        "Tomato target spot",

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
        "Tomato yellow leaf curl virus",

    "Tomato___Tomato_mosaic_virus":
        "Tomato mosaic virus",

    "Tomato___healthy":
        "Healthy tomato",
}


# =============================================================================
# CONFIGURATION
# =============================================================================

IMAGE_SIZE = 224

TOP_K_DEFAULT = 3

IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(
            236,
            antialias=True,
        ),

        transforms.CenterCrop(
            IMAGE_SIZE
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)

# A geometry-preserving transform used only to create an input-sized image for
# the qualitative heatmap overlay.
XAI_DISPLAY_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(
            236,
            antialias=True,
        ),

        transforms.CenterCrop(
            IMAGE_SIZE
        ),
    ]
)


# =============================================================================
# RESULT TYPES
# =============================================================================

@dataclass(frozen=True)
class ReliabilityPolicy:

    temperature: float
    confidence_threshold: float
    margin_threshold: float
    entropy_maximum: float

    validation_coverage: float | None = None
    validation_accepted_accuracy: float | None = None
    validation_accepted_balanced_accuracy: float | None = None
    validation_error_rejection_rate: float | None = None


@dataclass(frozen=True)
class RankedPrediction:

    rank: int
    class_name: str
    display_name: str
    probability: float


@dataclass(frozen=True)
class PredictionResult:

    crop: str

    class_name: str
    display_name: str

    confidence: float
    margin: float
    normalized_entropy: float

    accepted: bool

    confidence_pass: bool
    margin_pass: bool
    entropy_pass: bool

    policy: ReliabilityPolicy

    top_predictions: tuple[RankedPrediction, ...]

    original_width: int
    original_height: int

    model_input_width: int
    model_input_height: int

    device: str

    def as_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "crop":
                self.crop,

            "class_name":
                self.class_name,

            "display_name":
                self.display_name,

            "confidence":
                self.confidence,

            "margin":
                self.margin,

            "normalized_entropy":
                self.normalized_entropy,

            "accepted":
                self.accepted,

            "confidence_pass":
                self.confidence_pass,

            "margin_pass":
                self.margin_pass,

            "entropy_pass":
                self.entropy_pass,

            "policy":
                {
                    "temperature":
                        self.policy.temperature,

                    "confidence_threshold":
                        self.policy.confidence_threshold,

                    "margin_threshold":
                        self.policy.margin_threshold,

                    "entropy_maximum":
                        self.policy.entropy_maximum,

                    "validation_coverage":
                        self.policy.validation_coverage,

                    "validation_accepted_accuracy":
                        self.policy.validation_accepted_accuracy,

                    "validation_accepted_balanced_accuracy":
                        (
                            self.policy
                            .validation_accepted_balanced_accuracy
                        ),

                    "validation_error_rejection_rate":
                        self.policy.validation_error_rejection_rate,
                },

            "top_predictions":
                [
                    {
                        "rank":
                            item.rank,

                        "class_name":
                            item.class_name,

                        "display_name":
                            item.display_name,

                        "probability":
                            item.probability,
                    }

                    for item
                    in self.top_predictions
                ],

            "original_width":
                self.original_width,

            "original_height":
                self.original_height,

            "model_input_width":
                self.model_input_width,

            "model_input_height":
                self.model_input_height,

            "device":
                self.device,
        }


# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================

def _get_state_dict(
    checkpoint: Any,
) -> dict[str, torch.Tensor]:

    if isinstance(
        checkpoint,
        dict,
    ):

        for key in [
            "model_state_dict",
            "state_dict",
            "model",
        ]:

            candidate = checkpoint.get(
                key
            )

            if isinstance(
                candidate,
                dict,
            ):

                return candidate

        if (
            checkpoint
            and all(
                isinstance(
                    value,
                    torch.Tensor,
                )
                for value
                in checkpoint.values()
            )
        ):

            return checkpoint

    raise RuntimeError(
        "Could not locate model state_dict in checkpoint."
    )


def _strip_module_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:

    return {
        (
            key[
                len(
                    "module."
                ):
            ]
            if key.startswith(
                "module."
            )
            else key
        ):
        value

        for key, value
        in state_dict.items()
    }


# =============================================================================
# RELIABILITY HELPERS
# =============================================================================

def _normalized_entropy(
    probabilities: torch.Tensor,
) -> float:

    probabilities = probabilities.float()

    epsilon = 1e-12

    entropy = -torch.sum(
        probabilities
        * torch.log(
            probabilities
            + epsilon
        )
    )

    maximum_entropy = math.log(
        probabilities.numel()
    )

    return float(
        (
            entropy
            / maximum_entropy
        ).item()
    )


def _canonical_crop_name(
    crop: str,
) -> str:

    normalized = (
        str(
            crop
        )
        .strip()
        .lower()
    )

    aliases = {
        "tomato":
            "tomato",

        "tomatoes":
            "tomato",

        "lettuce":
            "lettuce",

        "lettuces":
            "lettuce",
    }

    if normalized not in aliases:

        raise ValueError(
            "Unsupported crop. Use 'tomato' or 'lettuce'."
        )

    return aliases[
        normalized
    ]


# =============================================================================
# ENGINE
# =============================================================================

class FinalPlantDiseaseEngine:

    def __init__(
        self,
        *,
        checkpoint_path: Path | str = CHECKPOINT_PATH,
        reliability_config_path: Path | str = RELIABILITY_CONFIG_PATH,
        device: torch.device | str = DEVICE,
    ) -> None:

        self.checkpoint_path = Path(
            checkpoint_path
        )

        self.reliability_config_path = Path(
            reliability_config_path
        )

        self.device = torch.device(
            device
        )

        self.model: nn.Module

        self.checkpoint_metadata: dict[str, Any]

        self.reliability_config: dict[str, Any]

        self.policies: dict[str, ReliabilityPolicy]

        self._load_reliability_config()

        self._load_model()

    # -------------------------------------------------------------------------
    # Reliability config
    # -------------------------------------------------------------------------

    def _load_reliability_config(
        self,
    ) -> None:

        if not self.reliability_config_path.is_file():

            raise FileNotFoundError(
                "Frozen reliability config not found:\n"
                f"{self.reliability_config_path}"
            )

        payload = json.loads(
            self.reliability_config_path.read_text(
                encoding="utf-8"
            )
        )

        for crop_name in [
            "tomato",
            "lettuce",
        ]:

            if crop_name not in payload:

                raise RuntimeError(
                    f"Reliability config is missing "
                    f"'{crop_name}'."
                )

        configured_checkpoint = (
            payload
            .get(
                "model",
                {}
            )
            .get(
                "checkpoint"
            )
        )

        if configured_checkpoint:

            configured_name = Path(
                str(
                    configured_checkpoint
                )
            ).name

            expected_name = (
                self.checkpoint_path.name
            )

            if (
                configured_name
                != expected_name
            ):

                raise RuntimeError(
                    "Frozen reliability config was produced for a "
                    "different checkpoint.\n\n"
                    f"Config : {configured_checkpoint}\n"
                    f"Model  : {self.checkpoint_path}"
                )

        policies = {}

        for crop_name in [
            "tomato",
            "lettuce",
        ]:

            config = payload[
                crop_name
            ]

            policies[
                crop_name
            ] = ReliabilityPolicy(
                temperature=float(
                    config[
                        "temperature"
                    ]
                ),

                confidence_threshold=float(
                    config[
                        "confidence_threshold"
                    ]
                ),

                margin_threshold=float(
                    config[
                        "margin_threshold"
                    ]
                ),

                entropy_maximum=float(
                    config[
                        "entropy_maximum"
                    ]
                ),

                validation_coverage=(
                    float(
                        config[
                            "validation_coverage"
                        ]
                    )
                    if (
                        config.get(
                            "validation_coverage"
                        )
                        is not None
                    )
                    else None
                ),

                validation_accepted_accuracy=(
                    float(
                        config[
                            "validation_accepted_accuracy"
                        ]
                    )
                    if (
                        config.get(
                            "validation_accepted_accuracy"
                        )
                        is not None
                    )
                    else None
                ),

                validation_accepted_balanced_accuracy=(
                    float(
                        config[
                            "validation_accepted_balanced_accuracy"
                        ]
                    )
                    if (
                        config.get(
                            "validation_accepted_balanced_accuracy"
                        )
                        is not None
                    )
                    else None
                ),

                validation_error_rejection_rate=(
                    float(
                        config[
                            "validation_error_rejection_rate"
                        ]
                    )
                    if (
                        config.get(
                            "validation_error_rejection_rate"
                        )
                        is not None
                    )
                    else None
                ),
            )

        self.reliability_config = payload

        self.policies = policies

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    def _load_model(
        self,
    ) -> None:

        if not self.checkpoint_path.is_file():

            raise FileNotFoundError(
                f"Checkpoint not found:\n"
                f"{self.checkpoint_path}"
            )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        state_dict = _strip_module_prefix(
            _get_state_dict(
                checkpoint
            )
        )

        model = convnext_tiny(
            weights=None
        )

        input_features = (
            model.classifier[
                2
            ].in_features
        )

        if (
            "classifier.2.1.weight"
            in state_dict
        ):

            output_classes = int(
                state_dict[
                    "classifier.2.1.weight"
                ].shape[
                    0
                ]
            )

            model.classifier = nn.Sequential(
                model.classifier[
                    0
                ],
                model.classifier[
                    1
                ],
                nn.Sequential(
                    nn.Dropout(
                        p=0.2
                    ),
                    nn.Linear(
                        input_features,
                        output_classes,
                    ),
                ),
            )

        elif (
            "classifier.2.weight"
            in state_dict
        ):

            output_classes = int(
                state_dict[
                    "classifier.2.weight"
                ].shape[
                    0
                ]
            )

            model.classifier = nn.Sequential(
                model.classifier[
                    0
                ],
                model.classifier[
                    1
                ],
                nn.Linear(
                    input_features,
                    output_classes,
                ),
            )

        else:

            raise RuntimeError(
                "Unknown ConvNeXt classifier layout."
            )

        if output_classes != 13:

            raise RuntimeError(
                f"Expected 13 checkpoint classes; "
                f"found {output_classes}."
            )

        model.load_state_dict(
            state_dict,
            strict=True,
        )

        model = model.to(
            self.device
        )

        model.eval()

        for parameter in model.parameters():

            parameter.requires_grad = False

        self.model = model

        self.checkpoint_metadata = (
            checkpoint
            if isinstance(
                checkpoint,
                dict,
            )
            else {}
        )

    # -------------------------------------------------------------------------
    # Public metadata
    # -------------------------------------------------------------------------

    def supported_crops(
        self,
    ) -> tuple[str, ...]:

        return (
            "tomato",
            "lettuce",
        )

    def class_names(
        self,
        crop: str,
    ) -> tuple[str, ...]:

        crop = _canonical_crop_name(
            crop
        )

        return tuple(
            CROP_CLASS_NAMES[
                crop
            ]
        )

    def policy(
        self,
        crop: str,
    ) -> ReliabilityPolicy:

        crop = _canonical_crop_name(
            crop
        )

        return self.policies[
            crop
        ]

    # -------------------------------------------------------------------------
    # Image preparation
    # -------------------------------------------------------------------------

    @staticmethod
    def _prepare_pil(
        image: Image.Image,
    ) -> Image.Image:

        return (
            ImageOps.exif_transpose(
                image
            )
            .convert(
                "RGB"
            )
        )

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------

    @torch.inference_mode()
    def predict_pil(
        self,
        image: Image.Image,
        *,
        crop: str,
        top_k: int = TOP_K_DEFAULT,
    ) -> PredictionResult:

        crop = _canonical_crop_name(
            crop
        )

        image = self._prepare_pil(
            image
        )

        original_width, original_height = (
            image.size
        )

        tensor = EVAL_TRANSFORM(
            image
        ).unsqueeze(
            0
        ).to(
            self.device
        )

        full_logits = self.model(
            tensor
        )

        checkpoint_indices = (
            CROP_CHECKPOINT_INDICES[
                crop
            ]
        )

        subset_logits = full_logits[
            :,
            checkpoint_indices,
        ]

        policy = self.policies[
            crop
        ]

        calibrated_logits = (
            subset_logits.float()
            / policy.temperature
        )

        probabilities = torch.softmax(
            calibrated_logits,
            dim=1,
        )[
            0
        ]

        class_names = CROP_CLASS_NAMES[
            crop
        ]

        number_of_classes = len(
            class_names
        )

        top_k = max(
            1,
            min(
                int(
                    top_k
                ),
                number_of_classes,
            ),
        )

        top_values, top_indices = torch.topk(
            probabilities,
            k=top_k,
        )

        top_values = (
            top_values.detach()
            .cpu()
            .tolist()
        )

        top_indices = (
            top_indices.detach()
            .cpu()
            .tolist()
        )

        confidence = float(
            top_values[
                0
            ]
        )

        if number_of_classes >= 2:

            probability_top2 = torch.topk(
                probabilities,
                k=2,
            ).values

            margin = float(
                (
                    probability_top2[
                        0
                    ]
                    - probability_top2[
                        1
                    ]
                ).item()
            )

        else:

            margin = 1.0

        entropy = _normalized_entropy(
            probabilities
        )

        confidence_pass = bool(
            confidence
            >= policy.confidence_threshold
        )

        margin_pass = bool(
            margin
            >= policy.margin_threshold
        )

        entropy_pass = bool(
            entropy
            <= policy.entropy_maximum
        )

        accepted = bool(
            confidence_pass
            and margin_pass
            and entropy_pass
        )

        leading_index = int(
            top_indices[
                0
            ]
        )

        leading_class = class_names[
            leading_index
        ]

        ranked = []

        for rank, (
            probability,
            class_index,
        ) in enumerate(
            zip(
                top_values,
                top_indices,
            ),
            start=1,
        ):

            class_name = class_names[
                int(
                    class_index
                )
            ]

            ranked.append(
                RankedPrediction(
                    rank=rank,

                    class_name=class_name,

                    display_name=DISPLAY_NAMES.get(
                        class_name,
                        class_name,
                    ),

                    probability=float(
                        probability
                    ),
                )
            )

        return PredictionResult(
            crop=crop,

            class_name=leading_class,

            display_name=DISPLAY_NAMES.get(
                leading_class,
                leading_class,
            ),

            confidence=confidence,

            margin=margin,

            normalized_entropy=entropy,

            accepted=accepted,

            confidence_pass=confidence_pass,

            margin_pass=margin_pass,

            entropy_pass=entropy_pass,

            policy=policy,

            top_predictions=tuple(
                ranked
            ),

            original_width=int(
                original_width
            ),

            original_height=int(
                original_height
            ),

            model_input_width=IMAGE_SIZE,

            model_input_height=IMAGE_SIZE,

            device=str(
                self.device
            ),
        )

    def predict_path(
        self,
        image_path: str | Path,
        *,
        crop: str,
        top_k: int = TOP_K_DEFAULT,
    ) -> PredictionResult:

        image_path = Path(
            image_path
        )

        if not image_path.is_file():

            raise FileNotFoundError(
                f"Image not found:\n"
                f"{image_path}"
            )

        with Image.open(
            image_path
        ) as source:

            image = self._prepare_pil(
                source
            )

            image.load()

        return self.predict_pil(
            image,
            crop=crop,
            top_k=top_k,
        )

    # -------------------------------------------------------------------------
    # Qualitative Grad-CAM++
    # -------------------------------------------------------------------------

    def gradcampp_pil(
        self,
        image: Image.Image,
        *,
        crop: str,
        target_class: str | None = None,
        overlay_alpha: float = 0.42,
    ) -> dict[str, Any]:

        """
        Generate a qualitative Grad-CAM++ attention map.

        Returns
        -------
        {
            "heatmap": np.ndarray HxW in [0, 1],
            "overlay": PIL.Image,
            "target_class": str,
            "target_display_name": str
        }

        IMPORTANT
        ---------
        This is an attention visualization, NOT lesion segmentation.
        """

        crop = _canonical_crop_name(
            crop
        )

        image = self._prepare_pil(
            image
        )

        display_image = XAI_DISPLAY_TRANSFORM(
            image
        )

        input_tensor = EVAL_TRANSFORM(
            image
        ).unsqueeze(
            0
        ).to(
            self.device
        )

        # XAI needs gradients temporarily.
        target_layer = self.model.features[
            7
        ][
            -1
        ]

        activations_holder = {}
        gradients_holder = {}

        def forward_hook(
            _module,
            _inputs,
            output,
        ):

            activations_holder[
                "value"
            ] = output

        def backward_hook(
            _module,
            _grad_input,
            grad_output,
        ):

            gradients_holder[
                "value"
            ] = grad_output[
                0
            ]

        forward_handle = target_layer.register_forward_hook(
            forward_hook
        )

        backward_handle = target_layer.register_full_backward_hook(
            backward_hook
        )

        try:

            # Temporarily enable autograd globally for the forward graph.
            with torch.enable_grad():

                input_tensor = input_tensor.detach()

                input_tensor.requires_grad_(
                    True
                )

                full_logits = self.model(
                    input_tensor
                )

                checkpoint_indices = (
                    CROP_CHECKPOINT_INDICES[
                        crop
                    ]
                )

                crop_logits = full_logits[
                    :,
                    checkpoint_indices,
                ]

                class_names = (
                    CROP_CLASS_NAMES[
                        crop
                    ]
                )

                if target_class is None:

                    policy = self.policies[
                        crop
                    ]

                    probabilities = torch.softmax(
                        crop_logits.float()
                        / policy.temperature,
                        dim=1,
                    )

                    target_index = int(
                        torch.argmax(
                            probabilities,
                            dim=1,
                        ).item()
                    )

                    target_class = class_names[
                        target_index
                    ]

                else:

                    if target_class not in class_names:

                        raise ValueError(
                            f"Class '{target_class}' is not "
                            f"a {crop} class."
                        )

                    target_index = class_names.index(
                        target_class
                    )

                score = crop_logits[
                    0,
                    target_index,
                ]

                self.model.zero_grad(
                    set_to_none=True
                )

                score.backward()

            activations = activations_holder.get(
                "value"
            )

            gradients = gradients_holder.get(
                "value"
            )

            if (
                activations is None
                or gradients is None
            ):

                raise RuntimeError(
                    "Could not capture Grad-CAM++ "
                    "activations/gradients."
                )

            activations = activations[
                0
            ]

            gradients = gradients[
                0
            ]

            gradient_squared = (
                gradients
                ** 2
            )

            gradient_cubed = (
                gradient_squared
                * gradients
            )

            spatial_activation_sum = torch.sum(
                activations,
                dim=(
                    1,
                    2,
                ),
                keepdim=True,
            )

            denominator = (
                2.0
                * gradient_squared
                +
                spatial_activation_sum
                * gradient_cubed
            )

            denominator = torch.where(
                torch.abs(
                    denominator
                )
                > 1e-12,

                denominator,

                torch.ones_like(
                    denominator
                ),
            )

            alphas = (
                gradient_squared
                / denominator
            )

            positive_gradients = torch.relu(
                gradients
            )

            weights = torch.sum(
                alphas
                * positive_gradients,
                dim=(
                    1,
                    2,
                ),
            )

            cam = torch.sum(
                weights[
                    :,
                    None,
                    None,
                ]
                * activations,
                dim=0,
            )

            cam = torch.relu(
                cam
            )

            cam = cam[
                None,
                None,
                :,
                :,
            ]

            cam = torch.nn.functional.interpolate(
                cam,
                size=(
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                mode="bilinear",
                align_corners=False,
            )[
                0,
                0,
            ]

            cam_min = torch.min(
                cam
            )

            cam_max = torch.max(
                cam
            )

            if float(
                (
                    cam_max
                    - cam_min
                ).item()
            ) > 1e-12:

                cam = (
                    cam
                    - cam_min
                ) / (
                    cam_max
                    - cam_min
                )

            else:

                cam = torch.zeros_like(
                    cam
                )

            heatmap = (
                cam.detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

            overlay = _make_heatmap_overlay(
                display_image,
                heatmap,
                alpha=float(
                    overlay_alpha
                ),
            )

            return {
                "heatmap":
                    heatmap,

                "overlay":
                    overlay,

                "target_class":
                    target_class,

                "target_display_name":
                    DISPLAY_NAMES.get(
                        target_class,
                        target_class,
                    ),

                "interpretation":
                    (
                        "Qualitative attention aid only; "
                        "not lesion segmentation."
                    ),
            }

        finally:

            forward_handle.remove()

            backward_handle.remove()


# =============================================================================
# HEATMAP OVERLAY
# =============================================================================

def _jet_like_rgb(
    heatmap: np.ndarray,
) -> np.ndarray:

    """
    Lightweight jet-like heatmap without matplotlib dependency.
    Input: HxW float [0, 1].
    Output: HxWx3 uint8.
    """

    x = np.clip(
        heatmap,
        0.0,
        1.0,
    )

    red = np.clip(
        1.5
        - np.abs(
            4.0
            * x
            - 3.0
        ),
        0.0,
        1.0,
    )

    green = np.clip(
        1.5
        - np.abs(
            4.0
            * x
            - 2.0
        ),
        0.0,
        1.0,
    )

    blue = np.clip(
        1.5
        - np.abs(
            4.0
            * x
            - 1.0
        ),
        0.0,
        1.0,
    )

    rgb = np.stack(
        [
            red,
            green,
            blue,
        ],
        axis=-1,
    )

    return (
        rgb
        * 255.0
    ).astype(
        np.uint8
    )


def _make_heatmap_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    *,
    alpha: float,
) -> Image.Image:

    image = (
        image
        .convert(
            "RGB"
        )
        .resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            Image.Resampling.LANCZOS,
        )
    )

    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    heatmap_rgb = _jet_like_rgb(
        heatmap
    ).astype(
        np.float32
    )

    alpha = float(
        np.clip(
            alpha,
            0.0,
            1.0,
        )
    )

    # Scale alpha by heat intensity so cold regions remain visually quiet.
    local_alpha = (
        alpha
        * heatmap[
            :,
            :,
            None,
        ]
    )

    overlay = (
        image_array
        * (
            1.0
            - local_alpha
        )
        +
        heatmap_rgb
        * local_alpha
    )

    overlay = np.clip(
        overlay,
        0.0,
        255.0,
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        overlay
    )


# =============================================================================
# CONVENIENCE SINGLETON
# =============================================================================

_ENGINE: FinalPlantDiseaseEngine | None = None


def get_engine() -> FinalPlantDiseaseEngine:

    global _ENGINE

    if _ENGINE is None:

        _ENGINE = FinalPlantDiseaseEngine()

    return _ENGINE


def predict_image(
    image: Image.Image,
    *,
    crop: str,
    top_k: int = TOP_K_DEFAULT,
) -> dict[str, Any]:

    return (
        get_engine()
        .predict_pil(
            image,
            crop=crop,
            top_k=top_k,
        )
        .as_dict()
    )


def predict_image_path(
    image_path: str | Path,
    *,
    crop: str,
    top_k: int = TOP_K_DEFAULT,
) -> dict[str, Any]:

    return (
        get_engine()
        .predict_path(
            image_path,
            crop=crop,
            top_k=top_k,
        )
        .as_dict()
    )


# =============================================================================
# SELF-TEST / CONSOLE SUMMARY
# =============================================================================

def _print_policy(
    crop: str,
    policy: ReliabilityPolicy,
) -> None:

    print(
        f"\n{crop.upper()}"
    )

    print(
        f"Temperature             : "
        f"{policy.temperature:.6f}"
    )

    print(
        f"Confidence threshold    : "
        f"{policy.confidence_threshold:.3f}"
    )

    print(
        f"Margin threshold        : "
        f"{policy.margin_threshold:.3f}"
    )

    print(
        f"Entropy maximum         : "
        f"{policy.entropy_maximum:.3f}"
    )

    if (
        policy.validation_coverage
        is not None
    ):

        print(
            f"Validation coverage     : "
            f"{100.0 * policy.validation_coverage:.2f}%"
        )

    if (
        policy.validation_accepted_accuracy
        is not None
    ):

        print(
            f"Validation accepted acc.: "
            f"{100.0 * policy.validation_accepted_accuracy:.2f}%"
        )


def main() -> None:

    print(
        "\n# Plant Disease AI — FINAL FROZEN MULTICROP INFERENCE ENGINE"
    )

    print(
        "=" * 112
    )

    engine = FinalPlantDiseaseEngine()

    print(
        f"Checkpoint              : "
        f"{engine.checkpoint_path}"
    )

    print(
        f"Reliability config      : "
        f"{engine.reliability_config_path}"
    )

    print(
        f"Device                  : "
        f"{engine.device}"
    )

    if engine.device.type == "cuda":

        print(
            f"GPU                     : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Checkpoint classes      : "
        f"{len(CHECKPOINT_CLASSES)}"
    )

    print(
        f"Tomato classes          : "
        f"{len(TOMATO_CLASSES)}"
    )

    print(
        f"Lettuce classes         : "
        f"{len(LETTUCE_CLASSES)}"
    )

    print(
        f"Model input             : "
        f"{IMAGE_SIZE} x {IMAGE_SIZE}"
    )

    _print_policy(
        "tomato",
        engine.policy(
            "tomato"
        ),
    )

    _print_policy(
        "lettuce",
        engine.policy(
            "lettuce"
        ),
    )

    print(
        "\n## DEPLOYMENT RULE"
    )

    print(
        "The app supplies crop identity (tomato or lettuce)."
    )

    print(
        "Uploaded images are internally resized/normalized automatically."
    )

    print(
        "A leading class is accepted only if its crop-specific reliability "
        "gate passes."
    )

    print(
        "\n## XAI RULE"
    )

    print(
        "Grad-CAM++ is a qualitative attention aid only."
    )

    print(
        "Do not describe it as accurate lesion localization or segmentation."
    )

    print(
        "\n# Engine initialized successfully."
    )


if __name__ == "__main__":

    main()
