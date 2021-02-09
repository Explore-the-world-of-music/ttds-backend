"""
This module creates the preprocessor class to load and preprocess the data
"""

from nltk.stem import PorterStemmer
import re
import os
import xml.etree.ElementTree as ET

# Set path as needed for Preprocessor class
path = os.path.abspath(__file__)
dname = os.path.dirname(os.path.dirname(path))
os.chdir(dname)

class Preprocessor():
    """
    Class to create preprocessor given the settings in the config
    """
    def __init__(self, config):
        self.stemmer = PorterStemmer()
        self.stopping = config["preprocessing"]["remove_stop_words"]
        self.stemming = config["preprocessing"]["use_stemming"]
        self.stop_set = set()
        with open("data/stopping_words.txt", "r") as stop:
            for word in stop:
                self.stop_set.add(word.rstrip())

    # Todo: Take out and replace by database (that is why it is implemented that inefficient)
    def load_data(self, file_path):
        with open(file_path, mode='rt', encoding='utf-8') as f:
            xml_root = ET.fromstring(f.read())
            num_docs = len(xml_root)
            doc_ids = [None] * num_docs
            raw_doc_texts = [None] * num_docs

            for idx, doc in enumerate(xml_root):
                doc_ids[idx] = int(doc.findtext('DOCNO'))
                raw_doc_texts[idx] = doc.findtext('HEADLINE') + doc.findtext('TEXT')
        return doc_ids, raw_doc_texts

    def load_data_from_db(self, model):
        song_ids =[]
        raw_lyrics = [] 
        for song in model.query.all():
            song_ids.append(song.id)
            raw_lyrics.append(song.lyrics)
        return song_ids, raw_lyrics

    def preprocess(self, line):
        """
        Function to perform the preprocessing for one line

        :param line: Raw input line (str)
        :return: Preprocessed line (str)
        """
        tokenized = re.findall("[\w]+", line)
        line = [x.lower() for x in tokenized if x != ""]
        # Todo: Combine stopping and stemming loop so that we only run once through lines
        if self.stopping:
            line = [x for x in line if x not in self.stop_set]
        if self.stemming:
            line = [self.stemmer.stem(x) for x in line]
        return line