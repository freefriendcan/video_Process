#!/usr/bin/env python3
"""Import the legacy ArcFace pickle gallery into the SQLite enrollment DB."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PipelineConfig  # noqa: E402
from repositories.face_repository import FaceRepository  # noqa: E402


def _load_pickle(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        loaded = pickle.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Gallery pickle must contain dict[str, np.ndarray]")

    gallery: dict[str, np.ndarray] = {}
    for label, embeddings in loaded.items():
        if not isinstance(label, str):
            raise ValueError("Gallery labels must be strings")
        gallery[label] = np.asarray(embeddings, dtype=np.float32)
    return gallery


def main() -> int:
    cfg = PipelineConfig()
    parser = argparse.ArgumentParser(description="Import data/embeddings/faces.pkl into faces.db")
    parser.add_argument("--input", default=cfg.gallery_path, type=Path)
    parser.add_argument("--db", default=cfg.face_db_path, type=Path)
    args = parser.parse_args()

    if not args.input.exists():
        logger.error("Gallery pickle not found: {}", args.input)
        return 1

    gallery = _load_pickle(args.input)
    repository = FaceRepository(args.db)
    repository.import_gallery(gallery)

    logger.success("Imported {} users into {}", len(gallery), args.db)
    for user in repository.list_users():
        logger.success("{}: {} embeddings", user.label, user.num_embeddings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
