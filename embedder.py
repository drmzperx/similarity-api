from __future__ import annotations
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Set

import numpy as np
import torch
import torch.nn as nn


def normalize_ingredient(name: str) -> str:
    """Lowercase, strip whitespace, replace spaces with underscores."""
    return name.strip().lower().replace(" ", "_")


def load_synonyms(synos_path: str) -> Dict[str, Set[str]]:
    """
    Build a bidirectional synonym lookup from inci_synos.jsonl.
    Only score=1.0 pairs are included.
    Returns: {ingredient: {synonym1, synonym2, ...}}
    """
    synonyms: Dict[str, Set[str]] = defaultdict(set)
    with open(synos_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("score") == 1.0:
                a = record["base_inci"]
                b = record["sim_inci"]
                synonyms[a].add(b)
                synonyms[b].add(a)
    return dict(synonyms)


def embed_ingredient(
    name: str,
    vocab: Dict[str, int],
    weights: np.ndarray,
) -> Optional[np.ndarray]:
    """Return the embedding vector for a single ingredient, or None if unknown."""
    key = normalize_ingredient(name)
    idx = vocab.get(key)
    if idx is None:
        return None
    return weights[idx]


def embed_list(
    csv_string: str,
    vocab: Dict[str, int],
    weights: np.ndarray,
    unknown_threshold: float = 0.5,
) -> Dict:
    """
    Embed a comma-separated ingredient list using position-weighted mean pooling.
    Weight of ingredient at position k (0-indexed): 1 / (k + 1)

    Note: position k is the ingredient's original list index. Unknown ingredients
    at earlier positions reduce the effective weight of later known ingredients.

    Returns dict with keys:
      embedding: np.ndarray
      known: list of known ingredient names
      unknown: list of unknown ingredient names
    Raises ValueError if list is empty or >unknown_threshold fraction are unknown.
    """
    tokens = [t.strip() for t in csv_string.split(",") if t.strip()]
    if not tokens:
        raise ValueError("Ingredient list is empty.")

    known, unknown = [], []
    total_weight = 0.0
    weighted_sum = np.zeros(weights.shape[1], dtype=np.float64)

    for k, token in enumerate(tokens):
        vec = embed_ingredient(token, vocab, weights)
        normalized = normalize_ingredient(token)
        if vec is None:
            unknown.append(normalized)
            print(f"Warning: unknown ingredient '{normalized}'", file=sys.stderr)
        else:
            w = 1.0 / (k + 1)
            weighted_sum += w * vec.astype(np.float64)
            total_weight += w
            known.append(normalized)

    unknown_ratio = len(unknown) / len(tokens)
    if unknown_ratio > unknown_threshold:
        raise ValueError(
            f"{len(unknown)}/{len(tokens)} ingredients are unknown "
            f"({unknown_ratio:.0%} > {unknown_threshold:.0%} threshold)."
        )

    if total_weight == 0.0:
        raise ValueError("No known ingredients found; cannot compute embedding.")

    embedding = (weighted_sum / total_weight).astype(np.float32)
    return {"embedding": embedding, "known": known, "unknown": unknown}


class Embedder:
    """
    Loads trained model artifacts and compares two ingredient lists.
    Requires inci_embeddings.pt, inci_regression.pt, and inci_synos.jsonl.
    """

    def __init__(
        self,
        embeddings_path: str,
        regression_path: str,
        synos_path: str,
        unknown_threshold: float = 0.5,
    ):
        for path in (embeddings_path, regression_path, synos_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Model artifact not found: {path}\n"
                    "Run python src/train.py first."
                )

        data = torch.load(embeddings_path, map_location="cpu", weights_only=False)
        self.vocab: Dict[str, int] = data["vocab"]
        self.weights: np.ndarray = data["weights"].numpy()

        self.reg_head = nn.Linear(1, 1)
        self.reg_head.load_state_dict(
            torch.load(regression_path, map_location="cpu", weights_only=True)
        )
        self.reg_head.eval()

        self.synonyms = load_synonyms(synos_path)
        self.unknown_threshold = unknown_threshold

    def compare(self, list_a: str, list_b: str) -> Dict:
        """
        Compare two comma-separated ingredient lists.
        Returns score + breakdown dict.
        """
        result_a = embed_list(list_a, self.vocab, self.weights,
                              unknown_threshold=self.unknown_threshold)
        result_b = embed_list(list_b, self.vocab, self.weights,
                              unknown_threshold=self.unknown_threshold)

        known_a = set(result_a["known"])
        known_b = set(result_b["known"])
        unknown = sorted(set(result_a["unknown"]) | set(result_b["unknown"]))

        shared = sorted(known_a & known_b)
        unique_a = sorted(known_a - known_b)
        unique_b = sorted(known_b - known_a)

        # Synonym matches: ingredient in unique_a that has a synonym in unique_b
        synonym_matches = []
        for ing_a in unique_a:
            syns = self.synonyms.get(ing_a, set())
            for ing_b in unique_b:
                if ing_b in syns:
                    synonym_matches.append((ing_a, ing_b))

        # Cosine similarity mapped to [0, 1] via (cos+1)/2
        emb_a = result_a["embedding"]
        emb_b = result_b["embedding"]
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        if norm_a == 0 or norm_b == 0:
            raw_cos = 0.0
        else:
            raw_cos = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
        cos_01 = (raw_cos + 1.0) / 2.0  # map [-1,1] -> [0,1]

        with torch.no_grad():
            cos_tensor = torch.tensor([[cos_01]], dtype=torch.float32)
            calibrated = float(self.reg_head(cos_tensor).item())
        score = float(np.clip(calibrated, 0.0, 1.0))

        return {
            "score": round(score, 4),
            "shared_ingredients": shared,
            "unique_to_a": unique_a,
            "unique_to_b": unique_b,
            "synonym_matches": synonym_matches,
            "unknown_ingredients": unknown,
        }
