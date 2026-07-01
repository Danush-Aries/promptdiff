"""`semantic_similarity` — reference-vs-output similarity with an offline fallback.

If ``sentence-transformers`` is installed and ``metric='cosine'`` is requested, the
embedding backend is used. Otherwise (and by default, and always in CI) a pure-Python
Jaccard token-overlap score keeps the assertion fully offline and dependency-free.
The backend is a swappable seam (``_backend``) so the embedding path is mockable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ..models import AssertionResult
from ._text import normalize_tokens
from .base import Assertion, AssertionConfigError, EvalContext

if TYPE_CHECKING:
    from ..models import AssertionSpec


def token_overlap(a: str, b: str) -> float:
    """Jaccard similarity over normalized token sets. Identical -> 1.0, disjoint -> 0.0."""
    ta, tb = set(normalize_tokens(a)), set(normalize_tokens(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def _embedding_backend(a: str, b: str) -> float:  # pragma: no cover - optional dep
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode([a, b], convert_to_tensor=True)
    return float(util.cos_sim(emb[0], emb[1]).item())


class SimilarityAssertion(Assertion):
    type_name = "semantic_similarity"

    def __init__(
        self,
        reference: str,
        threshold: float = 0.8,
        metric: str = "token_overlap",
        backend: Callable[[str, str], float] | None = None,
        negate: bool = False,
        description: str | None = None,
    ) -> None:
        super().__init__(negate=negate, description=description)
        if not 0.0 <= threshold <= 1.0:
            raise AssertionConfigError("semantic_similarity: threshold must be in [0, 1]")
        self.reference = reference
        self.threshold = threshold
        self.metric = metric
        self._backend = backend or self._resolve_backend(metric)

    @staticmethod
    def _resolve_backend(metric: str) -> Callable[[str, str], float]:
        if metric == "cosine":
            try:
                import sentence_transformers  # noqa: F401

                return _embedding_backend
            except ImportError:
                return token_overlap
        return token_overlap

    @classmethod
    def from_spec(cls, spec: AssertionSpec) -> SimilarityAssertion:
        cfg = spec.config
        reference = cfg.get("reference")
        if reference is None:
            raise AssertionConfigError("semantic_similarity: missing 'reference'")
        return cls(
            reference=str(reference),
            threshold=float(cfg.get("threshold", 0.8)),
            metric=str(cfg.get("metric", "token_overlap")),
            negate=spec.negate,
            description=spec.description,
        )

    def evaluate(self, output: str, ctx: EvalContext) -> AssertionResult:
        score = self._backend(output, self.reference)
        passed = score >= self.threshold
        backend_name = getattr(self._backend, "__name__", "custom")
        return AssertionResult(
            type=self.type_name,
            passed=passed,
            message=f"similarity {score:.3f} {'>=' if passed else '<'} {self.threshold:.3f}",
            detail={"score": score, "threshold": self.threshold, "backend": backend_name},
        )
