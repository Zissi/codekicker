from pytest import fixture

from codekicker.main import classified_sentences

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
def expected_classified_sentences():
    return {'cluster_1': ['Eine Fehlermeldung „Sie benötigen benötigen Adminrechte“ poppt auf.',
                'Ich habe Angst vor Spinnen'],
            'cluster_2': ['Der Versendeladebalken stoppt.']
            }

def test_classified_sentences(results, sentences, expected_classified_sentences, labels):
    assert classified_sentences(results, sentences, labels) == expected_classified_sentences