import nltk
import random
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import os


class PreprocessText():
    
    def __init__(self, files_pos, files_neg):
        self.files_pos = files_pos
        self.files_neg = files_neg

    def create_document(self):
        document = []
        try:
            load_doc = open("pickled_algos/documents.pickle","rb")
            document = pickle.load(load_doc)
            load_doc.close()
        except:
            for rev_pos, rev_neg in zip(self.files_pos, self.files_neg):
                document.append( (rev_pos, "pos"))
                document.append( (rev_neg, "neg"))

            #shuffle the document
            random.shuffle(document)
            # pickling the list documents to save future recalculations 
            save_documents = open("pickled_algos/documents.pickle","wb")
            pickle.dump(document, save_documents)
            save_documents.close()
            print("     Document Created.")
        return document

    def create_BOW(self, tagged_sentence):
        all_words = []
        allowed_word_types = ["J","R","V"]
        #allowed_word_types = ["J"]
        for w in tagged_sentence:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())
        return all_words

    def create_wordcloud(self, tagged_sentence, category):
        all_words = []
        #allowed_word_types = ["J","R","V"]
        allowed_word_types = ["J"]
        for w in tagged_sentence:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())
        text = ' '.join(all_words)
        wordcloud = WordCloud(height = 400, width = 800, background_color="white").generate(text)
        pathToSaveFile = './wordclouds/' + 'Wordcloud_'+ category +'.png'
        wordcloud.to_file(pathToSaveFile)


    def clean_tokenize_tag_wordcloud(self):
        stop_words = list(set(stopwords.words('english')))
        bow = []
        try:
            load_bow = open("pickled_algos/bow.pickle","rb")
            bow = pickle.load(load_bow)
            load_bow.close()
        except:
            print("No Saved BOW found. Continuing...")
            count = 0
            for rev_pos in self.files_pos:
                print("     Cleaning Review {}".format(count))
                count = count + 1
                cleaned = re.sub(r'[^(a-zA-Z)\s]','', rev_pos)
                tokenized = word_tokenize(cleaned)
                stopped = [w for w in tokenized if not w in stop_words]
                tagged_sentence = nltk.pos_tag(stopped)
                #self.create_wordcloud(tagged_sentence, "positive")
                bow = bow + self.create_BOW(tagged_sentence)

            for rev_neg in self.files_neg:
                print("     Cleaning Review {}".format(count))
                count = count + 1
                cleaned = re.sub(r'[^(a-zA-Z)\s]','', rev_neg)
                tokenized = word_tokenize(cleaned)
                stopped = [w for w in tokenized if not w in stop_words]
                tagged_sentence = nltk.pos_tag(stopped)
                #self.create_wordcloud(tagged_sentence, "negative")
                bow = bow + self.create_BOW(tagged_sentence)

            save_bow = open("pickled_algos/bow.pickle","wb")
            pickle.dump(bow, save_bow)
            save_bow.close()
        return bow

