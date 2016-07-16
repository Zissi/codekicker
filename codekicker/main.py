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


def read_sentences_and_labels_from_file(paths):
    labels = []
    sentences = []
    for path in paths:
        labels.append(os.path.basename(path))
        with open(path, 'r') as f:
            sentences.extend(f.readlines())
    return sentences, labels


def classified_sentences(results, sentences, labels):
    classified_sentences = defaultdict(list)
    for classification, sentence in zip(results, sentences):
        classified_sentences[labels[classification]].append(sentence)
    return classified_sentences


def main(paths):
    sentences, labels = read_sentences_and_labels_from_file(paths)
    features, vocabulary = extract_features_and_vocabulary(sentences)
    stemmed_target_features = [stem_words(target_feature) for target_feature in target_features]
    results = classify(stemmed_target_features, features, vocabulary)
    print(classified_sentences(results, sentences))
    print("Precission: %s" % sklearn.metrics.precision_score(labels, results))
    print("Recall: %s" % sklearn.metrics.recall_score(labels, results))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('paths', metavar="path", nargs='+', type=str,
                        help="")

    args = parser.parse_args()

    main(args.paths)