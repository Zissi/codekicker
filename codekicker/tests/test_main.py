import os

from pytest import fixture

from codekicker.main import classified_sentences, read_sentences_and_labels_from_file


@fixture
def sentences():
    return ['Eine Fehlermeldung „Sie benötigen benötigen Adminrechte“ poppt auf.',
            'Der Versendeladebalken stoppt.',
            'Ich habe Angst vor Spinnen']

@fixture
def results():
    return [0, 1, 0]

@fixture
def labels():
    return ['cluster_1', 'cluster_2']

@fixture
def paths():
    return [os.path.join(os.path.dirname(__file__), 'test_data/file_1'),
            os.path.join(os.path.dirname(__file__), 'test_data/file_2')]

@fixture
def expected_classified_sentences():
    return {'cluster_1': ['Eine Fehlermeldung „Sie benötigen benötigen Adminrechte“ poppt auf.',
                'Ich habe Angst vor Spinnen'],
            'cluster_2': ['Der Versendeladebalken stoppt.']
            }

def test_classified_sentences(results, sentences, expected_classified_sentences, labels):
    assert classified_sentences(results, sentences, labels) == expected_classified_sentences

def test_read_sentences_and_labels_from_file(paths):
    assert read_sentences_and_labels_from_file(paths) == (['Ein Satz.\n', 'Zwei Sätze', 'Drei Sätze!'], ['file_1', 'file_2'])