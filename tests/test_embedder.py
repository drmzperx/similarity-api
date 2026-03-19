import os
import pytest
import numpy as np
import torch

from embedder import (
    normalize_ingredient,
    load_synonyms,
    embed_ingredient,
    embed_list,
    Embedder,
)

# --- helpers -----------------------------------------------------------------

FAKE_VOCAB = {"aqua": 0, "glycerin": 1, "niacinamide": 2}
FAKE_WEIGHTS = np.array([
    [1.0, 0.0],  # aqua
    [0.0, 1.0],  # glycerin
    [0.5, 0.5],  # niacinamide
], dtype=np.float32)


def _make_fake_artifacts(tmp_path):
    """Write minimal fake model artifacts to tmp_path."""
    vocab = {"aqua": 0, "glycerin": 1, "niacinamide": 2, "vitamin_b3": 3}
    weights = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.7, 0.3],
        [0.7, 0.3],
    ])
    torch.save({"weights": weights, "vocab": vocab},
               tmp_path / "inci_embeddings.pt")

    reg = torch.nn.Linear(1, 1)
    with torch.no_grad():
        reg.weight.fill_(1.0)
        reg.bias.fill_(0.0)
    torch.save(reg.state_dict(), tmp_path / "inci_regression.pt")

    (tmp_path / "inci_synos.jsonl").write_text(
        '{"base_inci": "niacinamide", "sim_inci": "vitamin_b3", "score": 1.0}\n'
    )
    return tmp_path


# --- normalize_ingredient ----------------------------------------------------

class TestNormalizeIngredient:
    def test_lowercases(self):
        assert normalize_ingredient("Aqua") == "aqua"

    def test_strips_whitespace(self):
        assert normalize_ingredient("  glycerin  ") == "glycerin"

    def test_spaces_to_underscores(self):
        assert normalize_ingredient("sodium laureth sulfate") == "sodium_laureth_sulfate"

    def test_combined(self):
        assert normalize_ingredient("  Sodium Laureth Sulfate  ") == "sodium_laureth_sulfate"


# --- load_synonyms -----------------------------------------------------------

class TestLoadSynonyms:
    def test_bidirectional(self, tmp_path):
        f = tmp_path / "s.jsonl"
        f.write_text('{"base_inci": "niacinamide", "sim_inci": "vitamin_b3", "score": 1.0}\n')
        result = load_synonyms(str(f))
        assert "vitamin_b3" in result.get("niacinamide", set())
        assert "niacinamide" in result.get("vitamin_b3", set())

    def test_ignores_non_1_score(self, tmp_path):
        f = tmp_path / "s.jsonl"
        f.write_text('{"base_inci": "aqua", "sim_inci": "benzyl_alcohol", "score": 0.0}\n')
        result = load_synonyms(str(f))
        assert "benzyl_alcohol" not in result.get("aqua", set())


# --- embed_ingredient --------------------------------------------------------

class TestEmbedIngredient:
    def test_known_returns_vector(self):
        vec = embed_ingredient("aqua", FAKE_VOCAB, FAKE_WEIGHTS)
        assert vec is not None
        np.testing.assert_array_equal(vec, [1.0, 0.0])

    def test_unknown_returns_none(self):
        assert embed_ingredient("unknown_thing", FAKE_VOCAB, FAKE_WEIGHTS) is None

    def test_normalizes_before_lookup(self):
        assert embed_ingredient("  Aqua  ", FAKE_VOCAB, FAKE_WEIGHTS) is not None


# --- embed_list --------------------------------------------------------------

class TestEmbedList:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            embed_list("", FAKE_VOCAB, FAKE_WEIGHTS)

    def test_too_many_unknowns_raises(self):
        with pytest.raises(ValueError, match="unknown"):
            embed_list("x, y, z", FAKE_VOCAB, FAKE_WEIGHTS)

    def test_single_ingredient(self):
        result = embed_list("aqua", FAKE_VOCAB, FAKE_WEIGHTS)
        np.testing.assert_array_almost_equal(result["embedding"], [1.0, 0.0])

    def test_position_weighted_mean(self):
        # aqua weight=1.0, glycerin weight=0.5; total=1.5
        result = embed_list("aqua, glycerin", FAKE_VOCAB, FAKE_WEIGHTS)
        expected = (1.0 * np.array([1.0, 0.0]) + 0.5 * np.array([0.0, 1.0])) / 1.5
        np.testing.assert_array_almost_equal(result["embedding"], expected)

    def test_known_and_unknown_listed(self):
        result = embed_list("aqua, totally_unknown", FAKE_VOCAB, FAKE_WEIGHTS)
        assert "aqua" in result["known"]
        assert "totally_unknown" in result["unknown"]


# --- Embedder ----------------------------------------------------------------

class TestEmbedder:
    def test_loads_without_error(self, tmp_path):
        _make_fake_artifacts(tmp_path)
        e = Embedder(
            embeddings_path=str(tmp_path / "inci_embeddings.pt"),
            regression_path=str(tmp_path / "inci_regression.pt"),
            synos_path=str(tmp_path / "inci_synos.jsonl"),
        )
        assert e is not None

    def test_missing_artifact_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Embedder(
                embeddings_path=str(tmp_path / "missing.pt"),
                regression_path=str(tmp_path / "missing2.pt"),
                synos_path=str(tmp_path / "missing3.jsonl"),
            )

    def test_compare_score_in_range(self, tmp_path):
        _make_fake_artifacts(tmp_path)
        e = Embedder(
            embeddings_path=str(tmp_path / "inci_embeddings.pt"),
            regression_path=str(tmp_path / "inci_regression.pt"),
            synos_path=str(tmp_path / "inci_synos.jsonl"),
        )
        result = e.compare("aqua, glycerin", "aqua, glycerin")
        assert 0.0 <= result["score"] <= 1.0

    def test_compare_has_all_keys(self, tmp_path):
        _make_fake_artifacts(tmp_path)
        e = Embedder(
            embeddings_path=str(tmp_path / "inci_embeddings.pt"),
            regression_path=str(tmp_path / "inci_regression.pt"),
            synos_path=str(tmp_path / "inci_synos.jsonl"),
        )
        result = e.compare("aqua, niacinamide", "aqua, vitamin_b3")
        for key in ("score", "shared_ingredients", "unique_to_a",
                    "unique_to_b", "synonym_matches", "unknown_ingredients"):
            assert key in result

    def test_compare_detects_synonym(self, tmp_path):
        _make_fake_artifacts(tmp_path)
        e = Embedder(
            embeddings_path=str(tmp_path / "inci_embeddings.pt"),
            regression_path=str(tmp_path / "inci_regression.pt"),
            synos_path=str(tmp_path / "inci_synos.jsonl"),
        )
        result = e.compare("niacinamide", "vitamin_b3")
        assert len(result["synonym_matches"]) == 1
        assert set(result["synonym_matches"][0]) == {"niacinamide", "vitamin_b3"}

    def test_custom_unknown_threshold(self, tmp_path):
        _make_fake_artifacts(tmp_path)
        e = Embedder(
            embeddings_path=str(tmp_path / "inci_embeddings.pt"),
            regression_path=str(tmp_path / "inci_regression.pt"),
            synos_path=str(tmp_path / "inci_synos.jsonl"),
            unknown_threshold=0.0,  # reject any unknown ingredient
        )
        with pytest.raises(ValueError):
            e.compare("aqua, totally_unknown", "glycerin")
