import os

import numpy as np

import sklearn
from codekicker.classifier import TARGET_FEATURES, classify, predict_with_svc
from codekicker.evaluation import (evaluate_classification,
                                   evaluate_complete_classification)
from codekicker.preprocesser import (extract_features_and_vocabulary,
                                     stem_words, transform_to_tfidf)


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


def classify_with_expert_knowledge(paths):
    sentences, labels, class_names = load_test_data(paths)
    features, vocabulary, unused = extract_features_and_vocabulary(sentences)
    stemmed_target_features = [stem_words(target_feature) for target_feature in TARGET_FEATURES]
    predicted = classify(stemmed_target_features, features.toarray(), vocabulary)
    print('EXPERT KNOWLEDGE:')
    evaluate_classification(predicted, labels, sentences, class_names)


def classify_with_tf_idf(paths):
    sentences, labels, class_names = load_test_data(paths)
    train_sentences, test_sentences, train_labels, test_labels = sklearn.cross_validation.train_test_split(sentences,
                                                                                                           labels,
                                                                                                           test_size=0.2)
    train_features, vocabulary, count_vectorizer = extract_features_and_vocabulary(train_sentences)
    train_tfidf_features = transform_to_tfidf(train_features)
    clf = MultinomialNB().fit(train_tfidf_features, train_labels)
    test_features = extract_features_and_vocabulary_for_testing(test_sentences, count_vectorizer)
    test_tfidf_features = transform_to_tfidf(test_features)
    predicted = clf.predict(test_tfidf_features)

    print('TF-IDF')
    pprint(classified_sentences(predicted, sentences, class_names))
    print("Precission: %s" % sklearn.metrics.precision_score(test_labels, predicted))
    print("Recall: %s" % sklearn.metrics.recall_score(test_labels, predicted))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('paths', metavar="path", nargs='+', type=str,
                        help="")

    args = parser.parse_args()

    classify_with_expert_knowledge(args.paths)
    classify_with_tf_idf(args.paths)
