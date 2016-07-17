# codekicker

## Overview
This script reads the given paths and classifies the sentences found using two methods: [Expert knowledge](#expert_knowledge) and [Tf-idf](#tfidf).
For each method the average precision and recall are reported. For the tf-idf approach additionally the data is split into test and training set and the mean average precision and recall are reported.

## Setup
This is a python3 project. To install create a virtualenv (if wanted) and run `pip install -r requirements.txt`

## Usage
You can run the main script with `python main.py path1 path2 ...`, where `path1`, `path2` are the paths to the files containing sentences for the respective classes.
The file names will be used as class names. To get more information on the run command call the script with the `-h` switch.


## Methods used

### <a name="expert_knowledge"></a>Expert knowledge
Expert knowledge is used, that was given in the task description. The frequency of preselected distinctive words for technical support issues is counted.
 These words are hardcodeds in the script and are in German. The expert knowledge is refined by stemming these word, using the
 nltk Snowball stemmer. This makes them more usable for sentences, which do not contain these exact words. In comparison to the data set the expert knowledge is very extensive.


### <a name="tfidf"></a>Tf-idf (Term frequency - inverse document frequency)
For tf-idf the term frequency is counted and adjusted according to the frequency of the word in the document to reduce the noise
 introduces by very common words and increase the impact of rare words, which might carry more information.
  A Support Vector Machine classifier is trained on these adjusted word frequencies. SVM is a large margin classifier, which
 is commonly used if confidence is not needed. This method was added as an alternative that does not depend on expert knowledge,
  which can provide a classification for sentences, that do not contain any of the expert knowledge words.


## Results
The expert knowledge is quite extensive compared to the size of the data set, so the results from this method are already
very good, as would be expected. The precision is at over 0.84, which is double the baseline performance of only 0.42 precision.
 This is the precision that a naive classifier would have, by always guessing cluster_1, as the data set from the task is very unevenly distributed over the classes.
 For the same reason the precision for random guessing would be below 0.20.
 Tf-idf on its own performs much worse, with an mean average precision of only 0.49 (calculated over three-fold stratified cross-validation).
 This is only marginally better than the naive classifier. This is to be expected for such a small data set.
  However there might be advantage in combining the two methods, using tf-idf only if no expert knowledge applies.
Other mentioned possible improvements would be using the part of speech (POS) information, that was given with the expert knowledge.
 With this information it can be distinguished if the occurrence of the word from the expert knowledge has the same grammatical role in the
 classified sentence. This method is useful to reduce noise and get rid of ambiguity created by stemming. However in this very specific domain
 of technical support issued, the impact might be very small. Also the data set and expert knowledge are both very small, so further specification might be counter productive.