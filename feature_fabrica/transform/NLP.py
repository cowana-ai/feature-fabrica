# ngrams.py

import nltk
import numpy as np
from beartype import beartype
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.util import ngrams
from omegaconf import ListConfig
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from feature_fabrica.transform.base import Transformation
from feature_fabrica.transform.utils import NumericArray, StrArray, StrValue

nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')



class NGrams(Transformation):
    _name_ = "NGrams"
    @beartype
    def __init__(self, n: int):
        super().__init__()
        self.n = n

    @beartype
    def execute(self, data: StrArray | StrValue) -> StrArray:
        if isinstance(data, str):
            return np.array(['_'.join(gram) for gram in ngrams(data.split(), self.n)])
        else:
            return np.array([' '.join(['_'.join(gram) for gram in list(ngrams(text.split(), self.n))]) for text in data])


class Stemming(Transformation):
    _name_ = "Stemming"
    @beartype
    def __init__(self):
        super().__init__()
        self.stemmer = PorterStemmer()

    @beartype
    def execute(self, data: StrArray | StrValue) -> StrArray | StrValue:
        if isinstance(data, str):
            return ' '.join([self.stemmer.stem(word) for word in data.split()])
        else:
            return np.array([' '.join([self.stemmer.stem(word) for word in text.split()]) for text in data])


class Lemmatization(Transformation):
    _name_ = "Lemmatization"
    def __init__(self):
        super().__init__()
        self.lemmatizer = WordNetLemmatizer()

    @beartype
    def execute(self, data: StrArray | StrValue) -> StrArray | StrValue:
        if isinstance(data, str):
            return self._lemmatize_sentence(data)
        else:
            return np.array([self._lemmatize_sentence(text) for text in data])

    def _lemmatize_sentence(self, sentence: str) -> str:
        words = sentence.split()
        pos_tags = nltk.pos_tag(words)

        lemmatized_words = [
            self.lemmatizer.lemmatize(word, self._get_wordnet_pos(tag)) for word, tag in pos_tags
        ]
        return ' '.join(lemmatized_words)

    def _get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN


class TFIDF(Transformation):
    _name_ = "TFIDF"
    @beartype
    def __init__(self, max_features: int, ngram_range: tuple[int, int], stop_words: list[str] | None = None):
        super().__init__()
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words=stop_words)

    @beartype
    def execute(self, data: StrArray) -> NumericArray:
        return self.vectorizer.fit_transform(data).toarray()


class BagOfWords(Transformation):
    _name_ = 'BagOfWords'
    @beartype
    def __init__(self, max_features: int, ngram_range: tuple[int, int] | ListConfig):
        super().__init__()
        self.max_features = max_features

        # If ngram_range is a ListConfig, convert it to a tuple
        if isinstance(ngram_range, ListConfig):
            ngram_range = tuple(ngram_range)

        self.ngram_range = ngram_range
        self.vectorizer = CountVectorizer(
            max_features=self.max_features, ngram_range=self.ngram_range
        )

    @beartype
    def execute(self, data: StrArray) -> np.ndarray:
        return self.vectorizer.fit_transform(data).toarray()
