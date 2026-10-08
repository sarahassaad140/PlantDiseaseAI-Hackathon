# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 14:32:41 2026

@author: User
"""


from pathlib import Path
import hashlib
import urllib.request

ROOT = Path(__file__).resolve().parent

DEST = (
    ROOT
    / "checkpoints"
    / "convnext_tiny_targeted_repair_v1"
    / "best_balanced_accuracy.pth"
)

URL = (
    "https://github.com/sarahassaad140/"
    "PlantDiseaseAI-Hackathon/releases/download/"
    "v1.0.0-hackathon/best_balanced_accuracy.pth"
)

EXPECTED_SHA256 = (
    "b4f1ecf003f7710de4454f33babe841da9a302436a74a8a1b19590a68b3ea602"
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)

    if DEST.exists() and sha256(DEST) == EXPECTED_SHA256:
        print("Verified checkpoint already present.")
        return

    temporary = DEST.with_name(DEST.name + ".download")
    temporary.unlink(missing_ok=True)

    print("Downloading frozen ConvNeXt-Tiny checkpoint...")

    try:
        urllib.request.urlretrieve(URL, temporary)

        observed = sha256(temporary)
        if observed != EXPECTED_SHA256:
            raise RuntimeError(
                f"SHA-256 mismatch: {observed}"
            )

        temporary.replace(DEST)
    finally:
        temporary.unlink(missing_ok=True)

    print("Checkpoint downloaded and SHA-256 verified.")
    print("Saved to:", DEST)


if __name__ == "__main__":
    main()
