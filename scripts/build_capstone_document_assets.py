"""Build deterministic, repository-local capstone document artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from finai_academy.capstone.document_assets import load_certified_document_sources
from finai_academy.capstone.document_chunking import build_financial_chunks
from finai_academy.capstone.document_ingestion import PyMuPDF4LLMParser, render_evidence_crop
from finai_academy.capstone.document_models import (
    BoundingBox,
    DocumentElement,
    FinancialDocumentSource,
    ParsedDocument,
)

_MANIFEST_KEY = "assets/course-data/manifest.json"
_ELEMENTS_KEY = "assets/course-data/capstone/document_elements_v2.json"
_CHUNKS_KEY = "assets/course-data/capstone/financial_chunks_v2.json"
_NVIDIA_CROP_KEY = "assets/course-data/capstone/crops/nvidia_segment_table_page_165.png"
_SCHNEIDER_CROP_KEY = "assets/course-data/capstone/crops/schneider_revenue_tables_page_16.png"


class ArtifactContractError(ValueError):
    """Raised when the certified documents no longer meet artifact contracts."""


@dataclass(frozen=True)
class CapstoneArtifactBuild:
    """Safe summary of one complete offline-artifact build."""

    document_count: int
    page_count: int
    nvidia_target_table_count: int
    schneider_target_table_count: int
    artifact_sha256s: dict[str, str]


def build_capstone_document_assets(root: Path) -> CapstoneArtifactBuild:
    """Extract, chunk, render, and certify the complete tracked PDF corpus."""

    sources = load_certified_document_sources(root / _MANIFEST_KEY)
    parser = PyMuPDF4LLMParser()
    documents = tuple(parser.parse(source, project_root=root) for source in sources)
    chunks = tuple(chunk for document in documents for chunk in build_financial_chunks(document))

    _write_canonical_json(
        root / _ELEMENTS_KEY, [document.model_dump(mode="json") for document in documents]
    )
    _write_canonical_json(root / _CHUNKS_KEY, [chunk.model_dump(mode="json") for chunk in chunks])
    nvidia_tables, schneider_tables = _render_target_crops(sources, documents, root)
    artifact_sha256s = _artifact_sha256s(root)
    _write_manifest_record(root, sources, parser, artifact_sha256s)

    return CapstoneArtifactBuild(
        document_count=len(documents),
        page_count=sum(source.page_count for source in sources),
        nvidia_target_table_count=len(nvidia_tables),
        schneider_target_table_count=len(schneider_tables),
        artifact_sha256s=artifact_sha256s,
    )


def _write_canonical_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")


def _render_target_crops(
    sources: tuple[FinancialDocumentSource, ...],
    documents: tuple[ParsedDocument, ...],
    root: Path,
) -> tuple[tuple[DocumentElement, ...], tuple[DocumentElement, ...]]:
    sources_by_id = {source.document_id: source for source in sources}
    documents_by_id = {document.source.document_id: document for document in documents}
    nvidia = documents_by_id.get("NVDA-FY2026-ANNUAL-REPORT")
    schneider = documents_by_id.get("SU-FY2025-FULL-YEAR-RESULTS")
    if nvidia is None or schneider is None:
        raise ArtifactContractError("both certified documents are required")

    nvidia_tables = _target_tables(nvidia, page_number=165, shapes=((14, 4),))
    schneider_tables = _target_tables(
        schneider, page_number=16, shapes=((5, 6), (4, 6), (4, 6))
    )
    if len(nvidia_tables) != 1 or len(schneider_tables) != 3:
        raise ArtifactContractError("certified target table counts changed")

    render_evidence_crop(
        sources_by_id[nvidia.source.document_id],
        project_root=root,
        page_number=165,
        bbox=nvidia_tables[0].bbox,
        destination=root / _NVIDIA_CROP_KEY,
    )
    render_evidence_crop(
        sources_by_id[schneider.source.document_id],
        project_root=root,
        page_number=16,
        bbox=_combined_bbox(schneider_tables),
        destination=root / _SCHNEIDER_CROP_KEY,
    )
    return nvidia_tables, schneider_tables


def _target_tables(
    document: ParsedDocument, *, page_number: int, shapes: tuple[tuple[int, int], ...]
) -> tuple[DocumentElement, ...]:
    tables = tuple(
        element
        for element in document.elements
        if element.element_type == "table" and element.physical_page == page_number
    )
    actual_shapes = tuple(
        (element.table.row_count, element.table.column_count)
        for element in tables
        if element.table is not None
    )
    if actual_shapes != shapes:
        raise ArtifactContractError("certified target table shapes changed")
    return tables


def _combined_bbox(elements: tuple[DocumentElement, ...]) -> BoundingBox:
    return BoundingBox(
        x0=min(element.bbox.x0 for element in elements),
        y0=min(element.bbox.y0 for element in elements),
        x1=max(element.bbox.x1 for element in elements),
        y1=max(element.bbox.y1 for element in elements),
    )


def _artifact_sha256s(root: Path) -> dict[str, str]:
    keys = {
        "elements": _ELEMENTS_KEY,
        "chunks": _CHUNKS_KEY,
        "nvidia_crop": _NVIDIA_CROP_KEY,
        "schneider_crop": _SCHNEIDER_CROP_KEY,
    }
    return {name: sha256((root / key).read_bytes()).hexdigest() for name, key in keys.items()}


def _write_manifest_record(
    root: Path,
    sources: tuple[FinancialDocumentSource, ...],
    parser: PyMuPDF4LLMParser,
    artifact_sha256s: dict[str, str],
) -> None:
    manifest_path = root / _MANIFEST_KEY
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_paths = {
        "elements": _ELEMENTS_KEY,
        "chunks": _CHUNKS_KEY,
        "nvidia_crop": _NVIDIA_CROP_KEY,
        "schneider_crop": _SCHNEIDER_CROP_KEY,
    }
    manifest["capstone_derived_artifacts"] = [
        {
            "schema_version": 2,
            "parser": parser.parser_name,
            "parser_version": parser.parser_version,
            "chunking_strategy": "financial-context-v2",
            "source_sha256s": [source.sha256 for source in sources],
            **{
                name: {"path": path, "sha256": artifact_sha256s[name]}
                for name, path in artifact_paths.items()
            },
        }
    ]
    text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    manifest_path.write_text(text, encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    build = build_capstone_document_assets(root)
    print(
        "documents="
        f"{build.document_count} pages={build.page_count} "
        f"nvidia_target_tables={build.nvidia_target_table_count} "
        f"schneider_target_tables={build.schneider_target_table_count}"
    )


if __name__ == "__main__":
    main()
