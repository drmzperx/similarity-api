# Design: Incitrainer Model Integration

**Date:** 2026-03-19
**Status:** Approved

## Overview

Replace the current HuggingFace `SentenceTransformer` model (`drmzperx/inci-Hybrid-MiniLM-L6-V2`) with the locally trained incitrainer models. The new model compares two cosmetic product ingredient lists (1:1) and returns a structured similarity result with rich breakdown.

## Architecture

The API remains a single-file FastAPI app (`main.py`) augmented by a copied `embedder.py` module. Model artifacts are bundled directly in the repo under `models/`.

```
similarity-api/
├── main.py                     # updated: new Pydantic model, new endpoint logic
├── embedder.py                 # copied from incitrainer/src/embedder.py (adapted — see below)
├── models/
│   ├── inci_embeddings.pt      # ingredient vocab + 128-dim embeddings
│   ├── inci_regression.pt      # calibration regression head (Linear 1→1)
│   └── inci_synos.jsonl        # synonym pairs (score=1.0 only)
├── requirements.txt            # trimmed (see Dependencies section)
├── Dockerfile                  # remove HF install line; models/ already copied by existing COPY
└── docker-compose.yml          # unchanged
```

`inci_w2v.model` is not needed at inference and is not copied.

## Components

### `embedder.py`

Copied from `incitrainer/src/embedder.py` with the following adaptations — no other logic changes:

1. **Remove `import config`** at the top of the file.
2. **Update the `Embedder.__init__` signature** to accept `unknown_threshold: float = 0.5` as a parameter.
3. **Replace line** `self.unknown_threshold = config.UNKNOWN_THRESHOLD` **with** `self.unknown_threshold = unknown_threshold`.

These three changes make `embedder.py` self-contained with no cross-module imports.

Provides:
- `Embedder(embeddings_path, regression_path, synos_path, unknown_threshold=0.5)` — loads all artifacts on init
- `Embedder.compare(list_a, list_b)` — compares two comma-separated ingredient lists, returns score + breakdown

### `main.py`

**Startup:** `Embedder` is instantiated once at module load with hardcoded paths to `models/` relative to `main.py` (same pattern as current `SentenceTransformer` load).

```python
from embedder import Embedder
import os

BASE_DIR = os.path.dirname(__file__)
embedder = Embedder(
    embeddings_path=os.path.join(BASE_DIR, "models", "inci_embeddings.pt"),
    regression_path=os.path.join(BASE_DIR, "models", "inci_regression.pt"),
    synos_path=os.path.join(BASE_DIR, "models", "inci_synos.jsonl"),
)
```

**New Pydantic request model** (replaces the old `Query` with `query/corpus/authid`):
```python
class Query(BaseModel):
    list_a: str   # comma-separated ingredient list
    list_b: str   # comma-separated ingredient list
```

**Endpoint:** `POST /similarity/{domain}` — `domain` is logged but otherwise unused (preserved for caller compatibility).

**Error handling:** Wrap `embedder.compare()` in a try/except for `ValueError` and raise `HTTPException(status_code=422, detail=str(e))`. This covers both the >50%-unknown case and the empty-list case.

**Response:** the dict returned directly from `Embedder.compare()`:
```json
{
  "score": 0.8123,
  "shared_ingredients": ["glycerin"],
  "unique_to_a": ["niacinamide"],
  "unique_to_b": ["retinol"],
  "synonym_matches": [["water", "aqua"]],
  "unknown_ingredients": []
}
```

Note: `synonym_matches` is a list of 2-tuples in Python (`List[Tuple[str, str]]`); FastAPI serializes these as JSON arrays, so the wire format is `[["a", "b"], ...]`. No Pydantic response model is required — return the dict directly.

## Data Flow

```
POST /similarity/{domain}
  { list_a: "water, glycerin, ...", list_b: "aqua, glycerin, ..." }
        ↓
  embedder.compare(list_a, list_b)
        ↓
  embed_list(list_a) → position-weighted mean of known ingredient vectors
  embed_list(list_b) → position-weighted mean of known ingredient vectors
        ↓
  cosine similarity → mapped to [0,1] → regression head calibration
        ↓
  { score, shared_ingredients, unique_to_a, unique_to_b,
    synonym_matches, unknown_ingredients }
```

## Dependencies

### `requirements.txt`

Remove all packages that exist solely to support sentence-transformers and are unused by `embedder.py` or FastAPI:

**Remove:** `transformers`, `tokenizers`, `scipy`, `scikit-learn`, `pillow`, `tqdm`, `threadpoolctl`, `typer`, `requests`, `regex`

**Keep:** `fastapi[standard]`, `pydantic`, `uvicorn[standard]`, `numpy`, `typing_extensions`

Note: `sentence-transformers` is already commented out in `requirements.txt` — no action needed there.

`torch` stays in `requirements.txt` but is installed via the Dockerfile's dedicated CPU-wheel line (see below).

### `Dockerfile`

**Remove** this line (installs sentence-transformers and huggingface_hub):
```dockerfile
RUN pip install huggingface_hub sentence-transformers==3.3.1 --no-deps
```

**Keep** this line (forces CPU-only torch wheel, avoids large CUDA download):
```dockerfile
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**No new COPY line needed** — the existing `COPY ./ $APP_HOME` in the production stage already copies the entire project directory, including `models/`, into the image.

## Testing

- `POST /similarity/test` with two short ingredient lists → verify `score` is in `[0, 1]` and breakdown fields are present
- `GET /status` → update the displayed model name string to reflect the new model
- Error case — >50% unknown ingredients: send a list with mostly unknown names → verify HTTP 422 with descriptive message
- Error case — empty list: send `list_a: ""` → verify HTTP 422

## Out of Scope

- The `inci_w2v.model` training artifact
- Any 1:N ranking endpoint
- The `authid` field (removed — no longer relevant with the new input shape)
- The `/test/{text}` echo endpoint (unaffected; keep or remove independently)
