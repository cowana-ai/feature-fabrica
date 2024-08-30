from .base import Transformation
import numpy as np


class ToLower(Transformation):
    def execute(self, data: str) -> str:
        return data.lower()


class ToUpper(Transformation):
    def execute(self, data: str) -> str:
        return data.upper()


class Strip(Transformation):
    def __init__(self, chars: str | None = None):
        self.chars = chars

    def execute(self, data: str) -> str:
        return data.strip(self.chars)


class Split(Transformation):
    def __init__(self, delimiter: str):
        self.delimiter = delimiter

    def execute(self, data: str) -> list[str]:
        return data.split(self.delimiter)


class OneHotEncode(Transformation):
    def __init__(self, categories: list[str]):
        self.categories = categories
        self.category_map = {category: idx for idx, category in enumerate(categories)}

    def execute(self, data: str) -> np.ndarray:
        # Create a zero array with the length of categories
        one_hot = np.zeros(len(self.categories), dtype=int)
        # Set the index corresponding to the category to 1
        if data in self.category_map:
            one_hot[self.category_map[data]] = 1
        return one_hot


class LabelEncode(Transformation):
    def __init__(self, categories: list[str]):
        self.categories = categories
        self.category_map = {category: idx for idx, category in enumerate(categories)}

    def execute(self, data: str) -> int:
        # Return the index of the category or -1 if not found
        return self.category_map.get(data, -1)


class ExtractRegex(Transformation):
    def __init__(self, pattern: str):
        import re

        self.pattern = re.compile(pattern)

    def execute(self, data: str) -> str | None:
        match = self.pattern.search(data)
        return match.group(0) if match else None
