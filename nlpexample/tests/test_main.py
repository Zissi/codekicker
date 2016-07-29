import os

from nlpexample.main import load_test_data
from pytest import fixture


@fixture
def paths():
    return [os.path.join(os.path.dirname(__file__), 'test_data/file_1'),
            os.path.join(os.path.dirname(__file__), 'test_data/file_2')]


def test_read_sentences_and_labels_from_file(paths):
    expected_test_data =  (['Ein Satz.\n', 'Zwei Sätze', 'Drei Sätze!'], [0, 0, 1], ['file_1', 'file_2'])
    assert load_test_data(paths) == expected_test_data
