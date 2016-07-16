import numpy as np
from pytest import fixture

from codekicker.preprocesser import stem_words, extract_features_and_vocabulary


@fixture
def words():
    return ['benötigen']


@fixture
def stemmed_words():
    return ['benot']


@fixture
def input_sentences():
    return ['Eine Fehlermeldung „Sie benötigen benötigen Adminrechte“ poppt auf.',
            'Der Versendeladebalken stoppt.']

@fixture
def expected_vocabulary():
    return {'adminrecht': 0, 'auf': 1, 'benot': 2, 'der': 3, 'ein': 4, 'fehlermeld': 5, 'poppt': 6,
                           'sie': 7, 'stoppt': 8, 'versendeladebalk': 9}


def test_stem_words(words, stemmed_words):
    assert stem_words(words) == stemmed_words


def test_extract_features(input_sentences, expected_vocabulary):
    actual_features, actual_vocabulary = extract_features_and_vocabulary(input_sentences)
    expected_features = np.array([
        [1, 1, 2, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 1, 1]
    ])
    actual_vocabulary == expected_vocabulary
    assert (actual_features == expected_features).all()
