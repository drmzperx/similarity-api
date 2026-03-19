# Incitrainer Model Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the HuggingFace SentenceTransformer with the locally trained incitrainer `Embedder`, switching the `/similarity/{domain}` endpoint from 1:N ranking to 1:1 ingredient-list comparison.

**Architecture:** `embedder.py` (copied + adapted from incitrainer) provides the `Embedder` class; `main.py` imports it and loads model artifacts from `models/` at startup. Model files are bundled directly in the repo.

**Tech Stack:** Python 3.13, FastAPI, PyTorch (CPU), NumPy, pytest

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `models/inci_embeddings.pt` | Create (copy) | Ingredient vocab + embeddings |
| `models/inci_regression.pt` | Create (copy) | Calibration regression head |
| `models/inci_synos.jsonl` | Create (copy) | Synonym pairs |
| `embedder.py` | Create | Self-contained Embedder class (adapted from incitrainer) |
| `tests/test_embedder.py` | Create | Unit tests for Embedder and helpers |
| `tests/test_main.py` | Create | API integration tests via TestClient |
| `main.py` | Modify | New Query model, Embedder init, updated endpoint |
| `requirements.txt` | Modify | Remove dead sentence-transformers dependencies |
| `Dockerfile` | Modify | Remove HF install line |

---

## Task 1: Copy model artifacts

**Files:**
- Create: `models/inci_embeddings.pt`
- Create: `models/inci_regression.pt`
- Create: `models/inci_synos.jsonl`

- [ ] **Step 1: Copy the three model artifacts**

```bash
mkdir -p /home/ati/workspace/skinlyzer-ai/similarity-api/models
cp /home/ati/workspace/skinlyzer-ai/incitrainer/models/inci_embeddings.pt \
   /home/ati/workspace/skinlyzer-ai/similarity-api/models/
cp /home/ati/workspace/skinlyzer-ai/incitrainer/models/inci_regression.pt \
   /home/ati/workspace/skinlyzer-ai/similarity-api/models/
cp /home/ati/workspace/skinlyzer-ai/incitrainer/data/inci_synos.jsonl \
   /home/ati/workspace/skinlyzer-ai/similarity-api/models/
```

- [ ] **Step 2: Verify all three files are present**

```bash
ls -lh /home/ati/workspace/skinlyzer-ai/similarity-api/models/
```

Expected: three files listed — `inci_embeddings.pt`, `inci_regression.pt`, `inci_synos.jsonl`

- [ ] **Step 3: Add models/ to .gitignore or stage intentionally**

Check whether large `.pt` files should be committed or gitignored. If the team commits model files to the repo, stage them. If using git-lfs or external storage, add to `.gitignore` and document where to obtain them. For this project (files are small training outputs), commit them.

```bash
cd /home/ati/workspace/skinlyzer-ai/similarity-api
git add models/
git commit -m "Add incitrainer model artifacts to models/"
```

---

## Task 2: Create `embedder.py` (TDD)

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_embedder.py`
- Create: `embedder.py`

### Step 2a — Write the failing tests

- [ ] **Step 1: Create `tests/__init__.py`**

```bash
mkdir -p /home/ati/workspace/skinlyzer-ai/similarity-api/tests
touch /home/ati/workspace/skinlyzer-ai/similarity-api/tests/__init__.py
```

- [ ] **Step 2: Write `tests/test_embedder.py`**

Create `/home/ati/workspace/skinlyzer-ai/similarity-api/tests/test_embedder.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail (no `embedder.py` yet)**

```bash
cd /home/ati/workspace/skinlyzer-ai/similarity-api
pip install pytest --quiet
pytest tests/test_embedder.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'embedder'`

### Step 2b — Create `embedder.py`

- [ ] **Step 4: Create `embedder.py`** by copying from incitrainer and making the three spec-required adaptations.

Create `/home/ati/workspace/skinlyzer-ai/similarity-api/embedder.py` — identical to `/home/ati/workspace/skinlyzer-ai/incitrainer/src/embedder.py` EXCEPT:

1. Remove line 12: `import config`
2. Change the `__init__` signature from:
   ```python
   def __init__(self, embeddings_path: str, regression_path: str, synos_path: str):
   ```
   to:
   ```python
   def __init__(self, embeddings_path: str, regression_path: str, synos_path: str, unknown_threshold: float = 0.5):
   ```
3. Replace line 140: `self.unknown_threshold = config.UNKNOWN_THRESHOLD`
   with: `self.unknown_threshold = unknown_threshold`

No other changes.

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /home/ati/workspace/skinlyzer-ai/similarity-api
pytest tests/test_embedder.py -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
cd /home/ati/workspace/skinlyzer-ai/similarity-api
git add embedder.py tests/
git commit -m "Add embedder.py and unit tests (adapted from incitrainer)"
```

---

## Task 3: Update `main.py` (TDD)

**Files:**
- Create: `tests/test_main.py`
- Modify: `main.py`

### Step 3a — Write the failing API tests

- [ ] **Step 1: Write `tests/test_main.py`**

Create `/home/ati/workspace/skinlyzer-ai/similarity-api/tests/test_main.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/ati/workspace/skinlyzer-ai/similarity-api
pytest tests/test_main.py -v 2>&1 | head -40
```

Expected: several test failures. The model-name test and new-shape tests will fail because `main.py` still uses the old `SentenceTransformer` and old `Query` model.

### Step 3b — Update `main.py`

- [ ] **Step 3: Rewrite `main.py`**

Replace `/home/ati/workspace/skinlyzer-ai/similarity-api/main.py` with:

```python
import os
from typing import Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from embedder import Embedder

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
modelName = "incitrainer/inci-w2v-embedder"

embedder = Embedder(
    embeddings_path=os.path.join(BASE_DIR, "models", "inci_embeddings.pt"),
    regression_path=os.path.join(BASE_DIR, "models", "inci_regression.pt"),
    synos_path=os.path.join(BASE_DIR, "models", "inci_synos.jsonl"),
)


class Query(BaseModel):
    list_a: str
    list_b: str


@app.get("/")
def read_root():
    return {"Skinlyzer": "Similarity API v2"}


@app.post("/similarity/{domain}")
def similarity(domain: str, query: Query):
    print("Model: " + modelName)
    print("Domain: " + domain)

    try:
        result = embedder.compare(query.list_a, query.list_b)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return result


@app.get("/status")
def status():
    return {"status": "ok", "model": modelName}


@app.get("/test/{text}")
def echo(text: str):
    return {"text": text}
```

- [ ] **Step 4: Run all tests**

```bash
cd /home/ati/workspace/skinlyzer-ai/similarity-api
pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
cd /home/ati/workspace/skinlyzer-ai/similarity-api
git add main.py tests/test_main.py
git commit -m "Replace SentenceTransformer with incitrainer Embedder in main.py"
```

---

## Task 4: Trim dependencies and Dockerfile

**Files:**
- Modify: `requirements.txt`
- Modify: `Dockerfile`

- [ ] **Step 1: Update `requirements.txt`**

Replace the contents of `/home/ati/workspace/skinlyzer-ai/similarity-api/requirements.txt` with:

```
fastapi[standard]>=0.113.0,<0.114.0
pydantic>=2.7.0,<3.0.0
uvicorn[standard]
numpy
typing_extensions
torch
#sentence-transformers==3.3.1 --no-deps
```

Removed packages (no longer needed): `transformers`, `tokenizers`, `scipy`, `scikit-learn`, `pillow`, `tqdm`, `threadpoolctl`, `typer`, `requests`, `regex`

Note: `torch` stays in `requirements.txt` as a listed dependency. The Dockerfile's dedicated CPU-wheel `RUN` line ensures the CPU-only build is installed rather than the CUDA default — the `requirements.txt` entry is the declaration of the dependency, the Dockerfile line is the install strategy.

- [ ] **Step 2: Update `Dockerfile`**

Remove line 14 from `/home/ati/workspace/skinlyzer-ai/similarity-api/Dockerfile`:

```dockerfile
RUN pip install huggingface_hub sentence-transformers==3.3.1 --no-deps
```

Keep line 13 (CPU-only torch — this is load-bearing, do NOT remove):

```dockerfile
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The existing `COPY ./ $APP_HOME` line in the production stage already copies `models/` into the image — no new COPY line needed.

Final `Dockerfile` should look like:

```dockerfile
#Builder base
FROM python:3.13-slim AS builder
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV APP_HOME=/home/app
RUN mkdir -p $APP_HOME
WORKDIR $APP_HOME

ADD requirements.txt $APP_HOME
RUN pip install -r $APP_HOME/requirements.txt
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

#Production base
FROM python:3.13-slim

COPY --from=builder . .

COPY ./ $APP_HOME
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "1"]
```

- [ ] **Step 3: Run tests once more to confirm nothing broke**

```bash
cd /home/ati/workspace/skinlyzer-ai/similarity-api
pytest tests/ -v
```

Expected: all tests still PASS

- [ ] **Step 4: Commit**

```bash
cd /home/ati/workspace/skinlyzer-ai/similarity-api
git add requirements.txt Dockerfile
git commit -m "Trim sentence-transformers deps and remove HF Dockerfile install line"
```

---

## Done

At this point:
- All tests pass
- `main.py` uses the local `Embedder` with bundled model artifacts
- The old SentenceTransformer and HuggingFace dependencies are removed
- `POST /similarity/{domain}` accepts `{ list_a, list_b }` and returns the full ingredient comparison breakdown
