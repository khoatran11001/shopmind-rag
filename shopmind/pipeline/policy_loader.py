from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from shopmind.pipeline.knowledge_document import KnowledgeDocument


@dataclass(frozen=True)
class PolicySection:
    policy_id: str
    section_id: str
    title: str
    content: str
    policy_type: str
    version: str
    effective_date: str
    source_url: str | None
    metadata: dict[str, Any]

    @property
    def citation_id(self) -> str:
        return f"policy:{self.section_id}"

    @property
    def search_text(self) -> str:
        return f"{self.title}\n{self.content}".strip()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _section(raw: dict[str, Any], *, location: str) -> PolicySection:
    required = ("policy_id", "section_id", "title", "content", "policy_type", "version", "effective_date")
    missing = [key for key in required if not str(raw.get(key) or "").strip()]
    if missing:
        raise ValueError(f"{location} missing required fields: {', '.join(missing)}")
    policy_id = str(raw["policy_id"]).strip()
    section_id = str(raw["section_id"]).strip()
    if "." not in section_id:
        section_id = f"{policy_id}.{section_id}"
    if not section_id.startswith(f"{policy_id}."):
        raise ValueError(f"{location} section_id must start with {policy_id}.")
    metadata = dict(raw.get("metadata") or {})
    metadata["provenance"] = "shopmind_curated_experimental"
    return PolicySection(
        policy_id=policy_id,
        section_id=section_id,
        title=str(raw["title"]).strip(),
        content=str(raw["content"]).strip(),
        policy_type=str(raw["policy_type"]).strip(),
        version=str(raw["version"]).strip(),
        effective_date=str(raw["effective_date"]).strip(),
        source_url=None if raw.get("source_url") is None else str(raw["source_url"]),
        metadata=metadata,
    )


def load_policy_sections(path: str | Path) -> list[PolicySection]:
    path = Path(path)
    if path.is_dir():
        jsonl = path / "shopmind_policies.jsonl"
        if jsonl.exists():
            return load_policy_sections(jsonl)
        rows = []
        for yaml_path in sorted((*path.glob("*.yaml"), *path.glob("*.yml"))):
            raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            for section in raw.get("sections") or []:
                rows.append(_section({**section, "policy_id": raw["policy_id"],
                                      "policy_type": raw.get("policy_type", raw["policy_id"]),
                                      "version": raw["version"],
                                      "effective_date": raw["effective_date"],
                                      "source_url": raw.get("source_url")}, location=str(yaml_path)))
    else:
        rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError(f"{path}:{line_number} must contain a JSON object")
                rows.append(_section(raw, location=f"{path}:{line_number}"))
    ids = [section.section_id for section in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate policy section_id")
    return rows


def load_policy_corpus(path: str | Path) -> list[KnowledgeDocument]:
    return [KnowledgeDocument(
        document_id=section.citation_id,
        source="policy",
        title=section.title,
        content=section.content,
        search_text=section.search_text,
        metadata={"policy_id": section.policy_id, "section_id": section.section_id,
                  "policy_type": section.policy_type, "version": section.version,
                  "effective_date": section.effective_date, "source_url": section.source_url,
                  **section.metadata},
    ) for section in load_policy_sections(path)]
