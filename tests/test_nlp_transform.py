import unittest

import nltk
import numpy as np

from feature_fabrica.transform.NLP import (TFIDF, BagOfWords, Lemmatization,
                                           NGrams, Stemming)


class TestNPLTransformations(unittest.TestCase):
    def test_bag_of_words(self):
        bow_transform = BagOfWords(max_features=5, ngram_range=(1,1))
        data = np.array(["I love NLP", "NLP is awesome"])
        result = bow_transform.execute(data)
        expected_result = np.array([
            [0, 0, 1, 1],  # "I love NLP"
            [1, 1, 0, 1]  # "NLP is awesome"
        ])
        np.testing.assert_array_equal(result, expected_result)

    def test_tfidf_basic(self):
        data = np.array(["I love NLP", "NLP is awesome"])
        tfidf_transform = TFIDF(max_features=5, ngram_range=(1,1))
        result = tfidf_transform.execute(data)

        self.assertEqual(result.shape[0], 2)

    def test_stemming_basic(self):
        data = np.array(["running jumps"])
        stem_transform = Stemming()
        result = stem_transform.execute(data)

        expected_result = np.array(["run jump"])
        np.testing.assert_array_equal(result, expected_result)

    def test_lemmatization_basic(self):
        nltk.download('wordnet')
        data = np.array(["cats running"])
        lemma_transform = Lemmatization()
        result = lemma_transform.execute(data)

        expected_result = np.array(["cat run"])
        np.testing.assert_array_equal(result, expected_result)

    def test_ngrams_basic(self):
        data = np.array(["I love NLP"])
        ngram_transform = NGrams(2)
        result = ngram_transform.execute(data)

        # Expected bigrams: "I love" and "love NLP"
        expected_result = np.array(["I_love love_NLP"])
        np.testing.assert_array_equal(result, expected_result)

if __name__ == "__main__":
    unittest.main()
