"""Cross-encoder based reranker implementation."""

import asyncio

from pydantic import BaseModel, Field, InstanceOf
from sentence_transformers import CrossEncoder

from memmachine.common.utils import chunk_text, unflatten_like

from .reranker import Reranker


class CrossEncoderRerankerParams(BaseModel):
    """Parameters for CrossEncoderReranker."""

    cross_encoder: InstanceOf[CrossEncoder] = Field(
        ...,
        description="The cross-encoder model to use for reranking",
    )
    max_input_length: int | None = Field(
        default=None,
        description="Maximum input length for the model (in Unicode code points)",
        gt=0,
    )
    batch_size: int | None = Field(
        default=32,
        description="Batch size to use when calling cross-encoder predict",
        gt=0,
    )


class CrossEncoderReranker(Reranker):
    """Reranker that uses a cross-encoder model to score candidates."""

    def __init__(self, params: CrossEncoderRerankerParams) -> None:
        """Initialize a CrossEncoderReranker with provided parameters."""
        super().__init__()

        self._cross_encoder = params.cross_encoder
        self._max_input_length = params.max_input_length

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        """Score candidates for a query using the cross-encoder."""
        query = query[: self._max_input_length] if self._max_input_length else query

        chunked_candidates = [
            chunk_text(candidate_text, self._max_input_length)
            if self._max_input_length and candidate_text
            else [candidate_text]
            for candidate_text in candidates
        ]

        chunks = [
            chunk
            for candidate_chunks in chunked_candidates
            for chunk in candidate_chunks
        ]

        # predict in a thread with batching to improve throughput
        predict_fn = self._cross_encoder.predict
        inputs = [(query, chunk) for chunk in chunks]
        batch_size = (
            self._max_input_length
            if isinstance(self._max_input_length, int) and self._max_input_length > 0
            else 32
        )
        # if batch size provided in params, prefer that
        try:
            batch_size = (
                int(self._batch_size)
                if hasattr(self, "_batch_size") and getattr(self, "_batch_size")
                else batch_size
            )
        except Exception:
            pass

        results = []
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i : i + batch_size]
            batch_scores = await asyncio.to_thread(
                predict_fn, batch, show_progress_bar=False
            )
            results.extend([float(s) for s in batch_scores])
        chunk_scores = results

        chunked_candidate_scores = unflatten_like(chunk_scores, chunked_candidates)

        # Take the maximum score among chunks for each candidate.
        return [max(chunk_scores) for chunk_scores in chunked_candidate_scores]
