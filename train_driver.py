from trainer import TrainingTheModel
from preprocessor import PreprocessText
import os

class TrainDriver:
    def __init__(self):
        pass

    def driver(self):
        training_result = ""
        try:
            # Importing the dataset
            print("Reading Dataset POS")
            files_pos = os.listdir('dataset/train/pos')
            files_pos = [open('dataset/train/pos/'+f, 'r', encoding="utf8").read() for f in files_pos]
            print("Reading Dataset NEG")
            files_neg = os.listdir('dataset/train/neg')
            files_neg = [open('dataset/train/neg/'+f, 'r', encoding="utf8").read() for f in files_neg]

            # Preprocessing the Dataset
            print("PreProcessing Started.")
            print(" Cleaning Data.")
            pp = PreprocessText(files_pos, files_neg)
            documents = pp.create_document()
            print(" Creating Bag of Words")
            bow = pp.clean_tokenize_tag_wordcloud()
            print("PreProcessing Completed.")

            # Start training different classifiers
            tm = TrainingTheModel(bow)
            print(" Creating Frquent Words List.")
            frequent_words = tm.most_frequent_words()
            print(" Creating FeatureSets")
            featuresets = tm.create_featuresets(documents, frequent_words)
            print("Classifier Training Started.")
            tm.train_many_classifiers(featuresets)
            print(">>>>>>>>>>>>>>Training Completed<<<<<<<<<<<<<<<<<<")
            training_result = "Training Complete"

        except Exception as e:
            training_result = "Error occured in training: {}".format(str(e))
        
        return training_result