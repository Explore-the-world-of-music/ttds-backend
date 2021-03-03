import re
from collections import Counter
import dill
import os
import time
import pandas as pd

class Word_Completer():
    def __init__(self):
        """
        Initializes the Query_Completer class
        """
        self.token_counter = Counter()


    def tokenize_lyrics(self, lyrics):
        """
        Function which returns the tokenized and lowered lyrics

        :param lyrics: The lyrics to preprocess (str)
        :return line: Tokenized and lowered list of tokens (list)
        """
        tokenized = re.findall(r"[\w]+", lyrics.lower())

        line = [x for x in tokenized if x != ""]
        return line


    def add_single_lyric(self, lyrics):
        """
        Function which adds the tokens of the lyrics to the counter

        :param lyrics: The lyrics (str)
        """
        token_list = self.tokenize_lyrics(lyrics)
        self.token_counter.update(token_list)

    def predict_token(self, query, n):
        """
        Function which returns the n most probable tokens based on the last token of the query

        :param lyrics: The lyrics (str)
        """

        query_parts = query.lower().split()
        query_prior = " ".join(query_parts[:-1])
        query_relevant = query_parts[-1] # Extract the last token

        # retrieve all key which start with the given substring and are not exactly the substring (sorted by number of occurrences)
        sorted_counts = sorted([(key, value) for key, value in self.token_counter.items() if key.startswith(query_relevant) and key is not query_relevant], reverse= True, key=lambda x: x[1])
        
        # select the n most common tokens and return them
        n_results = [query_prior + " " + token for token, count in sorted_counts[0:n]]
        return n_results


    def save_model(self, model_filepath = "wc_model.pkl"):
        """
        Function which saved the model in the file

        :param model_filepath: Filepath to the model - has to be of type .pkl (str)
        """
        with open(model_filepath, 'wb') as file:
            dill.dump(self.token_counter, file)

    def load_model(self, model_filepath = "wc_model.pkl"):
        """
        Function which saved the model in the file

        :param model_filepath: Filepath to the model - has to be of type .pkl (str)
        """
        with open(model_filepath, 'rb') as file:
            self.token_counter = dill.load(file)


'''
# Set path as needed for Query_Completer class
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

wc = Word_Completer()

data = pd.read_csv("data-song_v1.csv")

begin = time.time()
for ids, (idx, row) in enumerate(data.iterrows()):
    if ids%1000 == 0:
        print(f"{ids} out of {data.shape[0]}", end='\r')
    wc.add_single_lyric(row["SongLyrics"])

print(f"Training took: {time.time() - begin}")


wc.save_model()
wc.load_model()

n = 5
begin = time.time()
print(wc.predict_token("Hell",n))
print(f"One prediction took: {time.time() - begin}")
print(wc.predict_token("World qua",n))
print(wc.predict_token("Oop",n))
print(wc.predict_token("Rip",n))
print(wc.predict_token("quat",n))'''

