import numpy as np

from codekicker.preprocesser import extract_features_and_vocabulary, stem_words
from pytest import fixture


@fixture
def words():
    return ['benötigen']


@fixture
def stemmed_words():
    return ['benot']


@fixture
def input_sentences():
    return ['Eine Fehlermeldung „Sie benötigen benötigen Adminrechte“ poppt auf.\n',
            'Der Versendeladebalken stoppt.']

@fixture
def expected_vocabulary():
    return {'adminrecht': 0, 'auf': 1, 'benot': 2, 'der': 3, 'ein': 4, 'fehlermeld': 5, 'poppt': 6,
                           'sie': 7, 'stoppt': 8, 'versendeladebalk': 9}


def test_stem_words(words, stemmed_words):
    assert stem_words(words) == stemmed_words


def test_extract_features(input_sentences, expected_vocabulary):
    actual_features, actual_vocabulary, unused = extract_features_and_vocabulary(input_sentences)
    expected_features = np.array([
        [1, 1, 2, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 1, 1]
    ])
    actual_vocabulary == expected_vocabulary
    assert (actual_features == expected_features).all()
