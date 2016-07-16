import os
from collections import defaultdict
from pprint import pprint

import sklearn

from codekicker.classifier import classify
from codekicker.preprocesser import extract_features_and_vocabulary, stem_words

target_features = [['versenden', 'verschicken', 'Email', 'Outlook', 'Thunderbird'],
                   ['installieren', 'Installation', 'Admin', 'Setup', 'Adminrechte'],
                   ['Maus', 'Mauszeiger', 'Zeiger', 'Cursor', 'Trackpad', 'Mousepad'],
                   ['Installation', 'Powerpoint', 'Computer', 'Excel', 'Formattierung', 'abstürzen'],
                   ['']]


def load_test_data(paths):
    labels = []
    class_names = []
    sentences = []
    for i, path in enumerate(paths):
        with open(path, 'r') as f:
            lines = f.readlines()
            sentences.extend(lines)
            class_names.append(os.path.basename(path))
            labels.extend([i] * len(lines))
    return sentences, labels, class_names


def classified_sentences(results, sentences, labels):
    classified_sentences = defaultdict(list)
    for classification, sentence in zip(results, sentences):
        classified_sentences[labels[classification]].append(sentence)
    return classified_sentences


def main(paths):
    sentences, labels, class_names = load_test_data(paths)
    features, vocabulary = extract_features_and_vocabulary(sentences)
    stemmed_target_features = [stem_words(target_feature) for target_feature in target_features]
    results = classify(stemmed_target_features, features, vocabulary)
    pprint(classified_sentences(results, sentences, class_names))
    print("Precission: %s" % sklearn.metrics.precision_score(labels, results))
    print("Recall: %s" % sklearn.metrics.recall_score(labels, results))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('paths', metavar="path", nargs='+', type=str,
                        help="")

    args = parser.parse_args()

    main(args.paths)