from __future__ import annotations

import json

from shopmind.app.llm.base import InvalidLLMOutput, LLMProviderError, LLMRequest, LLMResponse


class OpenAIProvider:
    def __init__(self, *, client=None, model: str = "gpt-5.4-mini", api_key: str | None = None) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise LLMProviderError("install the openai package to enable RAG generation") from exc
            client = OpenAI(api_key=api_key)
        self.client = client
        self.model = model

    def generate(self, request: LLMRequest) -> LLMResponse:
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "citation_ids": {"type": "array", "items": {"type": "string"}},
                "insufficient_evidence": {"type": "boolean"},
            },
            "required": ["answer", "citation_ids", "insufficient_evidence"],
            "additionalProperties": False,
        }
        try:
            response = self.client.responses.create(
                model=self.model,
                store=False,
                input=[
                    {"role": "system", "content": request.instructions},
                    {"role": "user", "content": f"Question: {request.question}\n\nEvidence (JSON lines):\n{request.rendered_context}\n\nAllowed citation IDs: {list(request.allowed_citation_ids)}"},
                ],
                text={"format": {"type": "json_schema", "name": request.schema_version, "schema": schema, "strict": True}},
            )
        except Exception as exc:
            raise LLMProviderError("answer generation is unavailable") from exc
        try:
            raw = json.loads(response.output_text)
            if not isinstance(raw, dict) or not isinstance(raw.get("answer"), str) or not isinstance(raw.get("citation_ids"), list) or not all(isinstance(item, str) for item in raw["citation_ids"]) or not isinstance(raw.get("insufficient_evidence"), bool):
                raise ValueError("invalid response shape")
            return LLMResponse(raw["answer"], tuple(raw["citation_ids"]), raw["insufficient_evidence"], {"provider": "openai", "model": getattr(response, "model", self.model), "response_id": getattr(response, "id", None)})
        except (ValueError, TypeError, AttributeError) as exc:
            raise InvalidLLMOutput("answer generation returned invalid structured output") from exc
