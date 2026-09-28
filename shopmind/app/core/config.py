from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ElasticsearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str
    index_alias: str = "products"
    username: str | None = None
    password: str | None = None


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str = "siglip2"
    model_name: str
    batch_size: int = Field(default=32, gt=0)
    hf_token: str | None = None


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    default_mode: str = "hybrid"
    top_k: int = Field(default=10, gt=0)
    candidate_k: int = Field(default=100, gt=0)


class FusionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: str = "rrf"
    rrf_k: int = Field(default=60, gt=0)


class RerankerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False


class ApiConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_image_bytes: int = Field(default=5 * 1024 * 1024, gt=0)


class RAGRetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    review_mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    policy_mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    product_candidates: int = Field(default=20, gt=0)
    review_candidates: int = Field(default=40, gt=0)
    policy_candidates: int = Field(default=20, gt=0)


class RAGFusionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Literal["rrf", "concat"] = "rrf"
    rrf_k: int = Field(default=60, gt=0)
    top_k: int = Field(default=40, gt=0)


class RAGRerankerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    model_name: str = "BAAI/bge-reranker-v2-m3"
    top_k: int = Field(default=12, gt=0)


class RAGContextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_documents: int = Field(default=12, gt=0)
    max_product: int = Field(default=4, ge=0)
    max_reviews: int = Field(default=5, ge=0)
    max_policies: int = Field(default=3, ge=0)
    max_characters: int = Field(default=24000, gt=0)


class RAGLLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["openai"] = "openai"
    model: str = "gpt-5.4-mini"
    api_key: str | None = None
    max_generation_attempts: int = Field(default=2, ge=1, le=2)


class RAGConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    retrieval: RAGRetrievalConfig = Field(default_factory=RAGRetrievalConfig)
    fusion: RAGFusionConfig = Field(default_factory=RAGFusionConfig)
    reranker: RAGRerankerConfig = Field(default_factory=RAGRerankerConfig)
    context: RAGContextConfig = Field(default_factory=RAGContextConfig)
    llm: RAGLLMConfig = Field(default_factory=RAGLLMConfig)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    elasticsearch: ElasticsearchConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    fusion: FusionConfig
    reranker: RerankerConfig
    api: ApiConfig = Field(default_factory=ApiConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)


def _overlay_env(data: dict[str, Any]) -> dict[str, Any]:
    elasticsearch = dict(data.get("elasticsearch") or {})
    embedding = dict(data.get("embedding") or {})
    env_mapping = {"ELASTICSEARCH_URL": (elasticsearch, "url"), "ELASTICSEARCH_USERNAME": (elasticsearch, "username"), "ELASTICSEARCH_PASSWORD": (elasticsearch, "password"), "HF_TOKEN": (embedding, "hf_token")}
    for env_name, (section, key) in env_mapping.items():
        value = os.getenv(env_name)
        if value is not None and value != "":
            section[key] = value
    data = dict(data)
    data["elasticsearch"] = elasticsearch
    data["embedding"] = embedding
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        rag = dict(data.get("rag") or {})
        rag["llm"] = {**(rag.get("llm") or {}), "api_key": api_key}
        data["rag"] = rag
    return data


def load_app_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError("application config root must be a mapping")
    return AppConfig.model_validate(_overlay_env(loaded))
