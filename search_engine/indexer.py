"""
This module creates the indexer class to create the index
"""

import re
import pickle
from collections import defaultdict
from tqdm import tqdm
from helpers.misc import create_default_dict_list
from array import *
from tqdm import tqdm

class Indexer:
    """
    Class to create indexer given the settings in the config
    """
    def __init__(self, config):
        self.index = None
        self.total_num_docs = None

    def build_index(self, preprocessor, doc_ids, raw_doc_texts):
        """
        Function to build the index based on given documents

        :param preprocessor: Preprocessor class instance (Preprocessor)
        :param doc_ids: List of document ids (list)
        :param raw_doc_texts: List of document texts (list)
        :return: index (dict)
        """
        # Initiate empty index dict
        index = defaultdict(create_default_dict_list)

        # Create dictionary entry for each token and doc id
        for doc_id, raw_line in tqdm(zip(doc_ids, raw_doc_texts), total = len(doc_ids)):

            if preprocessor.replacement_patterns:
                raw_line = preprocessor.replace_replacement_patterns(raw_line)
            line = preprocessor.preprocess(raw_line)
            for pos, token in enumerate(line):
                index[token][doc_id].append(pos)

        self.index = index
        return index

    # def add_all_doc_ids(self, doc_ids):
    #     self.all_doc_ids = [doc_id for doc_id in doc_ids]

    def store_index(self, as_pickle=True):
        """
        Function to save final index in a defined txt format
        """
        if as_pickle:
            with open('index.pickle', 'wb') as file:
                pickle.dump(self.index, file, protocol=pickle.HIGHEST_PROTOCOL)
        with open("index.txt", "w", encoding="utf8") as text_output:
            for term in sorted(self.index.keys()):
                text_output.write(f"{term}:{len(self.index[term])}\n")
                for doc in self.index[term]:
                    items = str(self.index[term][doc])[1:-1].replace(" ", "")
                    text_output.write(f"\t{doc}: {items}\n")

    def load_index(self, total_num_docs, as_pickle=True):
        """
        Function to load index from txt file
        :return: index (dict)
        """
        self.total_num_docs = total_num_docs
        if as_pickle:
            with open('index.pickle', 'rb') as file:
                index = pickle.load(file)
        else:
            # Initialise empty dictionary to store index
            index = {}
            with open('index.txt', "r", encoding="utf8") as f:
                for line in tqdm(f, total=47300935):
                    if '\t' not in line:
                        current_key = re.findall(r"[\w']+", line)[0]
                        index[current_key] = (array("I"), [])
                    else:
                        line = line.replace("\n", "").replace("\t", "").replace(" ", "")
                        document = int(line.split(":")[0])
                        index[current_key][0].append(document)
                        index[current_key][1].append(array("I", [int(x) for x in line.split(":")[1].split(",")]))
        return index
