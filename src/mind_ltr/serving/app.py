"""FastAPI serving layer — built together in Step 8.

GET /recommend?user_id=U123&k=10

Pipeline per request (mirrors Fig 6.23):
  retrieve ~200 candidates -> assemble features -> LightGBM rank
  -> MMR re-rank for diversity -> return top-k with scores.

Precomputed at startup ("feature store lite"):
  * item embeddings (FAISS index + numpy matrix)
  * item static features (parquet)
  * user history aggregates (parquet)
"""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="mind-ltr")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# TODO(step 8): /recommend endpoint
