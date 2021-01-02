import os
import preprocessor
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC
import nltk
import pickle
import random
from nltk.tokenize import word_tokenize


class TrainingTheModel():
    
    def __init__(self, bow):
        self.bow = bow

    def most_frequent_words(self):
        freq_bow = nltk.FreqDist(self.bow)
        frequent_words = list(freq_bow.keys())[:5000]
        print("Word Features Sample",frequent_words[0])
        save_frequent_words = open("pickled_algos/frequent_words5k.pickle","wb")
        pickle.dump(frequent_words, save_frequent_words)
        save_frequent_words.close()

        return frequent_words

    def find_features(self, rev, frequent_words):
        words = word_tokenize(rev)
        features = {}
        for w in frequent_words:
            features[w] = (w in words)

        return features

    def create_featuresets(self, documents, frequent_words):
        try:
            load_feats = open("pickled_algos/featset.pickle","rb")
            featuresets = pickle.load(load_feats)
            load_feats.close()
        except:
            featuresets = [(self.find_features(rev, frequent_words), category) for (rev, category) in documents] 
            random.shuffle(featuresets)
            print("FeatSet Length: " , len(featuresets))
            save_feats = open("pickled_algos/featset.pickle","wb")
            pickle.dump(featuresets, save_feats)
            save_feats.close()

        return featuresets

    def create_pickle(self, c, file_name):
        file_name = "pickled_algos/" + file_name + ".pickle" 
        save_classifier = open(file_name, 'wb')
        pickle.dump(c, save_classifier)
        save_classifier.close()

    def train_many_classifiers(self, featuresets):
        training_set = featuresets[:20000]
        testing_set = featuresets[20000:]
        print( 'training_set :', len(training_set), '\ntesting_set :', len(testing_set))
       
        print("ONB Classifier training started...")
        ONB_clf = nltk.NaiveBayesClassifier.train(training_set)
        print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(ONB_clf, testing_set))*100)
        self.create_pickle(ONB_clf, "ONB_clf")
        
        print("MNB Classifier training started...")
        MNB_clf = SklearnClassifier(MultinomialNB())
        MNB_clf.train(training_set)
        print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_clf, testing_set))*100)
        self.create_pickle(MNB_clf, "MNB_clf")

        print("BNB Classifier training started...")
        BNB_clf = SklearnClassifier(BernoulliNB())
        BNB_clf.train(training_set)
        print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BNB_clf, testing_set))*100)
        self.create_pickle(BNB_clf, "BNB_clf")

        print("LogReg Classifier training started...")
        LogReg_clf = SklearnClassifier(LogisticRegression())
        LogReg_clf.train(training_set)
        print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogReg_clf, testing_set))*100)
        self.create_pickle(LogReg_clf, "LogReg_clf")

        print("SGD Classifier training started...")
        SGD_clf = SklearnClassifier(SGDClassifier())
        SGD_clf.train(training_set)
        print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGD_clf, testing_set))*100)
        self.create_pickle(SGD_clf, "SGD_clf")

        print("SVC Classifier training started...")
        SVC_clf = SklearnClassifier(SVC())
        SVC_clf.train(training_set)
        print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_clf, testing_set))*100)
        self.create_pickle(SVC_clf, "SVC_clf")


