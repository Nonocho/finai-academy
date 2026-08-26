"""Certified source-asset manifest loading and integrity verification."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from finai_academy.capstone.document_models import FinancialDocumentSource


class SourceAssetError(ValueError):
    """Raised when a local certified source does not match its manifest record."""


def load_certified_document_sources(manifest_path: Path) -> tuple[FinancialDocumentSource, ...]:
    """Load the complete official documents certified in a course manifest."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return tuple(
        FinancialDocumentSource.model_validate(item) for item in payload["capstone_documents"]
    )


def verify_source_asset(source: FinancialDocumentSource, root: Path) -> None:
    """Fail closed when a certified document differs from its recorded bytes."""

    path = root / source.local_asset_key
    raw = path.read_bytes()
    if len(raw) != source.byte_size:
        raise SourceAssetError("certified document byte size mismatch")
    if sha256(raw).hexdigest() != source.sha256:
        raise SourceAssetError("certified document SHA-256 mismatch")
