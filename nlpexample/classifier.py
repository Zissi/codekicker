from nlpexample.preprocesser import (extract_features_for_testing,
                                     transform_to_tfidf)
from sklearn.svm import LinearSVC

TARGET_FEATURES = [['versenden', 'verschicken', 'Email', 'Outlook', 'Thunderbird'],
                   ['installieren', 'Installation', 'Admin', 'Setup', 'Adminrechte'],
                   ['Maus', 'Mauszeiger', 'Zeiger', 'Cursor', 'Trackpad', 'Mousepad'],
                   ['Installation', 'Powerpoint', 'Computer', 'Excel', 'Formattierung', 'abstürzen'],
                   []]

def get_frequency_in_sentence(word, sentence_features, vocabulary):
    try:
        feature_index = vocabulary[word]
        return sentence_features[feature_index]
    except KeyError:
        return 0


def sum_target_feature_frequencies(target_feature, sentence_features, vocabulary):
    return sum(get_frequency_in_sentence(word, sentence_features, vocabulary) for word in target_feature)


def classify_sentence(target_features, sentence_features, vocabulary):
    target_feature_frequencies = [sum_target_feature_frequencies(target_feature, sentence_features, vocabulary) for
                                  target_feature in target_features]
    if max(target_feature_frequencies):
        return target_feature_frequencies.index(max(target_feature_frequencies))
    else:
        # The last target class does not have any target features.
        # It contains all sentences, which do not contain any of the known target features.
        return len(target_features) - 1


def classify(target_features, features, vocabulary):
    classification = [classify_sentence(target_features, sentence_feature, vocabulary) for sentence_feature in features]
    return classification


def predict_with_svc(tfidf_features_train, labels_train, sentences_test, count_vectorizer):
    clf = LinearSVC().fit(tfidf_features_train, labels_train)
    features_test = extract_features_for_testing(sentences_test, count_vectorizer)
    tfidf_features_test = transform_to_tfidf(features_test)
    return clf.predict(tfidf_features_test)
