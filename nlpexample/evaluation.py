from collections import defaultdict
from pprint import pprint

import numpy as np

import sklearn


def classified_sentences(results, sentences, labels):
    classified_sentences = defaultdict(list)
    for classification, sentence in zip(results, sentences):
        classified_sentences[labels[classification]].append(sentence)
    return classified_sentences


def evaluate_classification(predicted, labels, sentences, class_names):
    pprint(classified_sentences(predicted, sentences, class_names))
    precision = sklearn.metrics.precision_score(labels, predicted, average=None, labels=range(5))
    recall = sklearn.metrics.recall_score(labels, predicted, average=None, labels=range(5))
    average_precision = np.mean(precision)
    average_recall = np.mean(recall)
    print("Precision: %s" % precision)
    print("Average precision: %s" % average_precision)
    print("Recall: %s" % recall)
    print("Average recall: %s" % average_recall)
    return average_precision, average_recall

def evaluate_complete_classification(average_precisions, average_recalls,):
    mean_average_precision = np.mean(average_precisions)
    mean_average_recall = np.mean(average_recalls)
    print('Mean average precision: %s' % mean_average_precision)
    print('Mean average recall: %s' % mean_average_recall)
