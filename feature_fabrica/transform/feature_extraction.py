from collections.abc import Callable, Iterable, Mapping

from beartype import beartype
from sklearn.feature_extraction.text import CountVectorizer

from feature_fabrica.transform.base import Transformation
from feature_fabrica.transform.utils import NumericArray, StrArray


class CountVectorize(Transformation):
    @beartype
    def __init__(self, vocabulary: Mapping | Iterable | None = None, input: str = 'content', encoding: str = 'utf-8', lowercase: bool = True,
                 tokenizer: Callable | None = None, analyzer: str | Callable = 'word', **kwargs):
        super().__init__()
        self.vectorizer = CountVectorizer(vocabulary=vocabulary, input=input, encoding=encoding,
                                       lowercase=lowercase, tokenizer=tokenizer,
                                       analyzer=analyzer, **kwargs)
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.analyzer = analyzer

    @beartype
    def execute(self, data: StrArray) -> NumericArray:
        # If vocabulary is provided, no need to fit
        token_matrix = self.vectorizer.transform(data) if self.vocabulary else self.vectorizer.fit_transform(data)

        return token_matrix.toarray()  # Return as a NumPy array
