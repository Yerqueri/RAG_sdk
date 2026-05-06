from typing import List

from .base_entity_extraction_strategy import BaseEntityExtractionStrategy

# spaCy type → our canonical type
_TYPE_MAP = {
    "PERSON": "PERSON",
    "PER": "PERSON",
    "ORG": "ORG",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "FAC": "LOCATION",
    "PRODUCT": "PRODUCT",
    "EVENT": "EVENT",
    "WORK_OF_ART": "CONCEPT",
    "LAW": "CONCEPT",
    "LANGUAGE": "CONCEPT",
    "NORP": "ORG",
}


class SpacyEntityExtractionStrategy(BaseEntityExtractionStrategy):
    """
    Fast, offline NER using spaCy. Extracts entity names and types but
    does NOT extract semantic relationships between entities.

    Requires:
        pip install spacy
        python -m spacy download en_core_web_sm
    """

    def __init__(self, model: str = "en_core_web_sm"):
        try:
            import spacy  # noqa: F401
        except ImportError:
            raise ImportError(
                "spaCy is not installed. Run: pip install spacy && "
                "python -m spacy download en_core_web_sm"
            )
        import spacy as _spacy
        self.nlp = _spacy.load(model)

    def extract(self, text: str) -> List[dict]:
        if not text or not text.strip():
            return []
        doc = self.nlp(text[:5000])  # cap to avoid OOM on very large chunks
        seen = set()
        results = []
        for ent in doc.ents:
            if ent.text in seen:
                continue
            seen.add(ent.text)
            entity_type = _TYPE_MAP.get(ent.label_, "OTHER")
            results.append({"entity": ent.text, "type": entity_type, "relations": []})
        return results

