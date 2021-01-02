import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

class EnsembleClassifier(ClassifierI):
    
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

class EnsembleBuilder:
    
    def __init__(self):
        pass

    def load_documents(self):
        documents_f = open("pickled_algos/documents.pickle", "rb")  
        documents = pickle.load(documents_f)
        documents_f.close()
        return documents
    
    def load_frequent_words(self):
        word_features5k_f = open("pickled_algos/frequent_words5k.pickle", "rb")
        word_features = pickle.load(word_features5k_f)
        word_features5k_f.close()
        return word_features


    def load_model(self, file_path):
        classifier_f = open(file_path, "rb")
        classifier = pickle.load(classifier_f)
        classifier_f.close()
        return classifier

    def predict_ensemble(self, features):
        ONB_Clf = self.load_model('pickled_algos/ONB_clf.pickle')
        MNB_Clf = self.load_model('pickled_algos/MNB_clf.pickle')
        BNB_Clf = self.load_model('pickled_algos/BNB_clf.pickle')
        LogReg_Clf = self.load_model('pickled_algos/LogReg_clf.pickle')
        SGD_Clf = self.load_model('pickled_algos/SGD_clf.pickle')

        ensemble_clf = EnsembleClassifier(ONB_Clf, MNB_Clf, BNB_Clf, LogReg_Clf, SGD_Clf)   
        result = []
        classified = ensemble_clf.classify(features)
        result.append(classified)
        confidence = ensemble_clf.confidence(features)
        result.append(confidence)
        return result

    def parse_text(self, document):
        words = word_tokenize(document)
        features = {}
        word_features = self.load_frequent_words()
        for w in word_features:
            features[w] = (w in words)
        return features

    def make_prediction(self, text):
        features = self.parse_text(text)
        result = self.predict_ensemble(features)
        revcat = ""
        if result[0] == "pos":
            revcat = "POSITIVE"
        elif result[0] == "neg":
            revcat = "NEGATIVE"
        print("Result = {}".format(result))
        output_string = "Your Review is {} with a Score of {}%.".format(revcat, result[1]*100)
        return output_string