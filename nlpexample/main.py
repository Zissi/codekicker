import os

import numpy as np

import sklearn
from nlpexample.classifier import TARGET_FEATURES, classify, predict_with_svc
from nlpexample.evaluation import (evaluate_classification,
                                   evaluate_complete_classification)
from nlpexample.preprocesser import (extract_features_and_vocabulary,
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
    sentences = np.array(sentences)
    labels = np.array(labels)
    average_precisions = []
    average_recalls = []
    for train_index, test_index in sklearn.cross_validation.StratifiedKFold(labels, n_folds=3):
        sentences_train, sentences_test = sentences[train_index], sentences[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]
        features_train, vocabulary, count_vectorizer = extract_features_and_vocabulary(sentences_train)
        tfidf_features_train = transform_to_tfidf(features_train)
        predicted = predict_with_svc(tfidf_features_train, labels_train, sentences_test, count_vectorizer)

        print('TF-IDF')
        average_precision, average_recall = evaluate_classification(predicted, labels_test, sentences, class_names)
        average_precisions.append(average_precision)
        average_recalls.append(average_recall)
    evaluate_complete_classification(average_precisions, average_recalls)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="This script reads the given paths and classifies the sentences found there"
                                                 " in two different ways. For each method the average precision and recall"
                                                 " are reported.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('paths', metavar="path", nargs='+', type=str,
                        help="Path to a file containing sentences for a single class. "
                             "Filename will be used as class name")
    args = parser.parse_args()

    classify_with_expert_knowledge(args.paths)
    classify_with_tf_idf(args.paths)
