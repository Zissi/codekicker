import nltk
from sklearn.feature_extraction.text import CountVectorizer


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
    return count_vectorizer.fit_transform(sentences).toarray(), count_vectorizer.vocabulary_
