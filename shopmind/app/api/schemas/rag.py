from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class RAGAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    product_id: str | None = Field(default=None, max_length=128)
    sources: list[Literal["product", "review", "policy"]] = Field(default_factory=lambda: ["product", "review", "policy"], min_length=1)
    debug: bool = False

    @field_validator("question", "product_id")
    @classmethod
    def strip_nonempty(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()


class RAGSourceResponse(BaseModel):
    document_id: str
    source: str
    title: str
    score: float
    product_id: str | None = None
    policy_version: str | None = None
    source_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGAskResponse(BaseModel):
    answer: str
    sources: list[RAGSourceResponse]
    insufficient_evidence: bool
    metadata: dict[str, Any] = Field(default_factory=dict)
