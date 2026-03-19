import pytest
from fastapi.testclient import TestClient

# Import app — this triggers Embedder init at module load.
# Requires models/inci_embeddings.pt, models/inci_regression.pt,
# and models/inci_synos.jsonl to be present (copied in Task 1).
from main import app

client = TestClient(app)


class TestRootEndpoint:
    def test_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_contains_skinlyzer_key(self):
        response = client.get("/")
        assert "Skinlyzer" in response.json()


class TestStatusEndpoint:
    def test_returns_ok(self):
        response = client.get("/status")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_model_name_is_incitrainer(self):
        response = client.get("/status")
        assert response.json().get("model") == "incitrainer/inci-w2v-embedder"


class TestSimilarityEndpoint:
    def test_valid_request_returns_200(self):
        response = client.post(
            "/similarity/test",
            json={"list_a": "water, glycerin", "list_b": "aqua, glycerin"},
        )
        assert response.status_code == 200

    def test_response_has_score(self):
        response = client.post(
            "/similarity/test",
            json={"list_a": "water, glycerin", "list_b": "aqua, glycerin"},
        )
        body = response.json()
        assert "score" in body
        assert 0.0 <= body["score"] <= 1.0

    def test_response_has_all_breakdown_keys(self):
        response = client.post(
            "/similarity/test",
            json={"list_a": "water, glycerin", "list_b": "aqua, niacinamide"},
        )
        body = response.json()
        for key in ("score", "shared_ingredients", "unique_to_a",
                    "unique_to_b", "synonym_matches", "unknown_ingredients"):
            assert key in body, f"Missing key: {key}"

    def test_empty_list_a_returns_422(self):
        response = client.post(
            "/similarity/test",
            json={"list_a": "", "list_b": "water, glycerin"},
        )
        assert response.status_code == 422

    def test_empty_list_b_returns_422(self):
        response = client.post(
            "/similarity/test",
            json={"list_a": "water, glycerin", "list_b": ""},
        )
        assert response.status_code == 422

    def test_mostly_unknown_ingredients_returns_422(self):
        response = client.post(
            "/similarity/test",
            json={
                "list_a": "xyzunknown1, xyzunknown2, xyzunknown3",
                "list_b": "water",
            },
        )
        assert response.status_code == 422

    def test_old_request_shape_rejected(self):
        # Old shape (query + corpus + authid) should NOT work
        response = client.post(
            "/similarity/test",
            json={
                "query": "water, glycerin",
                "corpus": ["aqua, glycerin"],
                "authid": ["id1"],
            },
        )
        # Should fail (422) because list_a and list_b are required
        assert response.status_code == 422
