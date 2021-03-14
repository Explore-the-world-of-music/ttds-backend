"""
This module creates the preprocessor class to load and preprocess the data
"""

from nltk.stem import PorterStemmer
import re
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pickle

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
        self.replacement_patterns = config["preprocessing"]["use_replacement_patterns"]
        self.stop_set = set()
        with open("data/stopping_words.txt", "r") as stop:
            for word in stop:
                self.stop_set.add(word.rstrip())
        with open("data/replacement_patterns.txt", "r") as patterns:
            self.list_replacement_patterns = [tuple(pattern.rstrip().split(",")) for pattern in patterns]

    def load_data_from_db(self, song_model, artist_model):
        song_ids =[]
        data = []
        rows = song_model.query.count()

        for song in tqdm(song_model.query.join(artist_model).all(), total = rows):
            song_ids.append(song.id)

            song_data = ""
            if song.artist.name is not None:
                song_data = song_data + " " + song.artist.name  # Add artist name
            if song.name is not None:
                song_data = song_data + " " + song.name  # Add song name
            if song.album is not None:
                song_data = song_data + " " + song.album  # Add song album
            if song.genre is not None:
                song_data = song_data + " " + song.genre  # Add song genre
            song_data = song_data + " " + song.lyrics  # Add song lyrics

            data.append(song_data)

        with open("data.pickle", "wb") as data_file:
            pickle.dump((song_ids, data), data_file)
                
        return song_ids, data

    def replace_replacement_patterns(self, line):
        """
        Function to replace the patterns given by an defined replacement, e.g. it's to it is

        :param line: Input line (str)
        :return: Processed input line (str)
        """
        for (pattern, replacement) in self.list_replacement_patterns:
            cur_replace_regex = re.compile(pattern, re.IGNORECASE)
            line = cur_replace_regex.sub(replacement, line)
        return line

    def preprocess(self, line):
        """
        Function to perform the preprocessing for one line

        :param line: Raw input line (str)
        :return: Preprocessed line (str)
        """
        # Replace \\n by \n which we need for the data loading
        line = line.replace("\\n", "\n")

        # Tokenize, remove stop words and perform stemming
        tokenized = re.findall("[\w]+", line.lower())
        line = [x for x in tokenized if x != ""]
        if self.stopping:
            line = [x for x in line if x not in self.stop_set]
        if self.stemming:
            line = [self.stemmer.stem(x) for x in line]
        return line