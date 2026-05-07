from abc import ABC, abstractmethod
from typing import List


class BaseEntityExtractionStrategy(ABC):
    @abstractmethod
    def extract(self, text: str) -> List[dict]:
        """
        Extract named entities and relationships from *text*.

        Returns a list of entity dicts, each shaped as:
        {
            "entity":    str,           # entity surface form
            "type":      str,           # PERSON | ORG | LOCATION | CONCEPT | PRODUCT | EVENT | OTHER
            "relations": [              # may be empty
                {"target": str, "relation": str},
                ...
            ]
        }
        """
        pass

