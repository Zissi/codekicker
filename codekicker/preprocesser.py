import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class StemmingTokenizer(object):
    def __init__(self):
        self.tokenizer = nltk.RegexpTokenizer(r'\w+')

    def __call__(self, doc):
        words = self.tokenizer.tokenize(doc)
        return stem_words(words)


def stem_words(words):
    stemmer = nltk.stem.snowball.GermanStemmer(ignore_stopwords=False)
    return [stemmer.stem(word) for word in words]


def extract_features_and_vocabulary(sentences):
    count_vectorizer = CountVectorizer(analyzer="word", tokenizer=StemmingTokenizer(), preprocessor=None)
    return count_vectorizer.fit_transform(sentences), count_vectorizer.vocabulary_, count_vectorizer

def extract_features_and_vocabulary_for_testing(sentences, count_vectorizer):
    return count_vectorizer.transform(sentences)

def transform_to_tfidf(features):
    tfidf_transformer = TfidfTransformer()
    return tfidf_transformer.fit_transform(features)
