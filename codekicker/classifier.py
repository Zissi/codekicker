from codekicker.preprocesser import extract_features_and_vocabulary, stem_words


def get_frequency_in_sentence(word, sentence_features, vocabulary):
    try:
        feature_index = vocabulary[word]
    except KeyError:
        return 0
    return sentence_features[feature_index]


def sum_target_feature_frequencies(target_feature, sentence_features, vocabulary):
    return sum(get_frequency_in_sentence(word, sentence_features, vocabulary) for word in target_feature)


def classify_sentence(target_features, sentence_features, vocabulary):
    target_feature_frequencies = [sum_target_feature_frequencies(target_feature, sentence_features, vocabulary) for
                                  target_feature in target_features]
    if max(target_feature_frequencies):
        return target_feature_frequencies.index(max(target_feature_frequencies))
    else:
        # The last target class contains all sentences, which have no target features in comman with any target class.
        return len(target_features) - 1


def classify(target_features, features, vocabulary):
    classification = [classify_sentence(target_features, sentence_feature, vocabulary) for sentence_feature in features]
    return classification
