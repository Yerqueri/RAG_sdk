import json
import re
from typing import List

from .base_entity_extraction_strategy import BaseEntityExtractionStrategy

_EXTRACTION_PROMPT = """You are a precise knowledge-graph builder.
Extract all named entities and relationships from the text below.

Return ONLY a valid JSON array (no markdown, no explanation). Each element must have:
  "entity"    – the entity name (string)
  "type"      – one of: PERSON, ORG, LOCATION, CONCEPT, PRODUCT, EVENT, OTHER
  "relations" – array of {{"target": <entity_name>, "relation": <verb_phrase>}}

If there are no entities, return an empty array: []

Text:
{text}
"""


class LLMEntityExtractionStrategy(BaseEntityExtractionStrategy):
    """
    Uses the already-configured LLM to extract entities and relationships
    from text via a structured JSON prompt.
    """

    def __init__(self, llm_provider: str = None):
        # Import here to avoid circular imports at module load time
        from rag_sdk.factories.llm_factory import LLMFactory
        self.llm = LLMFactory.get_llm(provider=llm_provider)

    # ------------------------------------------------------------------ #

    def extract(self, text: str) -> List[dict]:
        if not text or not text.strip():
            return []

        prompt = _EXTRACTION_PROMPT.format(text=text[:3000])  # cap to avoid token overrun
        try:
            response = self.llm.invoke(prompt)
            raw = response.content if hasattr(response, "content") else str(response)
            return self._parse_json(raw)
        except Exception as exc:
            print(f"[EntityExtractor] LLM call failed: {exc}")
            return []

    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_json(raw: str) -> List[dict]:
        """Robustly extract a JSON array from potentially noisy LLM output."""
        # Strip markdown code fences
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

        # Try direct parse first
        try:
            result = json.loads(raw)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        # Fall back: find the first [...] block
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        print(f"[EntityExtractor] Could not parse LLM output as JSON: {raw[:200]}")
        return []

