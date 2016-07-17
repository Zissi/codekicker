# codekicker

## Overview
This script reads the given paths and classifies the sentences found using two methods: [Expert knowledge](#expert_knowledge) and [Tf-idf](#tfidf).
For each method the average precision and recall are reported. For the tf-idf approach the data is split into test and training set and the mean average precision and recall are reported.

## setup
This is a python3 project. To install create a virtualenv (if wanted) and run `pip install -r requirements.txt`

## Usage:
You can run the main script with `python main.py path1 path2 ...`, where `path1`, `path2` are the paths to the files containing sentences for the respective classes.
The file names will be used as class names. To get more information on the run command call the script with the `-h` switch.


## Methods used

### <a name="expert_knowledge"></a>Expert knowledge
Expert knowledge is used, that was given in the task description. For this by counting known distinctive words for technical support issues.
 These words are fixed in the script and are in German. The expert knowledge is refined by stemming the words in it using the
 nltk Snowball stemmer, which makes them more usable for sentences, which do not contain the exact word. In comparison to the data set the expert knowledge is very extensive.


### <a name="tfidf"></a>Tf-idf Term frequency - in document frequency
In tf idf the term frequency is counted and adjusted according to the document frequency, to reduce the noise of very common words
and increase the impact of rare words, which might carry more information. A Support Vector Machine classifier is trained on this
 adjusted word frequencies. For this the data is split 3 times into test and trainings set. SVM is a large margin classifier, which
 is commonly used if confidence is not needed. This method was added to try out a different approach, than the expert knowledge, and also to
 provide a classification for sentences, which do not contain any of the expert knowledge words.


## Results
The expert knowledge is quite extensive compared to the size of the data set, so the results from this method are already
very good, as would be expected. The precision is at over 0.84, which is double the baseline performance of only 0.42 precision.
 This is the precision that a naive classifier would have, by always guessing cluster_1, as the data set from the task is very unevenly distributed.
 The precision for random guessing would therefore be below 0.20. Tf-idf on its own performs much worse, with an mean average precision of only 0.49.
 This is only marginally better than the naive classifier. This is to be expected for such a small data set.
  However there might be advantage in combining the too methods, only using tf-idf if no expert knowledge applies.
Other mentioned possible improvements would be using the part of speech (POS) information, that was given with the expert knowledge.
 With this information it can be distinguished if the occurrence of the word from the expert knowledge has the same grammatical role in the
 classified sentence. This method is useful to reduce noise and get rid of ambiguity created by stemming. However in this very specific domain
 of technical support issued, the impact might be very small. Also the data set and expert knowledge are both very small, so further specification might be contra productive.