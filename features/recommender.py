from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
import random
import dill
import time

class RecommendationEngine():
    def __init__(self, vector_size = 100, learning_rate = 0.0025, min_learning_rate = 0.0000025, min_count = 1):
        self.model = Doc2Vec(vector_size = vector_size, alpha = learning_rate, min_alpha= min_learning_rate, min_count = min_count)
        self.contained_doc_ids = {}

    def train(self, tokenized_lyrics_list, song_ids, max_epochs):

        '''
        Trains the model again - based on the provided samples

        :param tokenized_lyrics_list: List of Lists of preprocessed, tokenized lyrics (str)
        :param tokenized_lyrics_list: List of corresponding song IDs (int)
        :params max_epochs: Number of maximum training epochs (int)
        '''

        # Bring Documents into relevant format for Doc2Vec
        # Can be simply a list of elements, but for larger corpora,consider an iterable that streams the documents directly from disk/network.
        tagged_data = []
        for tokenized_lyrics, song_id in zip(tokenized_lyrics_list, song_ids):
            tagged_data.append(TaggedDocument(words=tokenized_lyrics, tags=[str(song_id)]))
            self.contained_doc_ids[song_id] = True

        self.model.build_vocab(tagged_data)

        # save learning rate and determine linear decay rate
        prior_learning_rate = self.model.alpha
        learning_decay = prior_learning_rate / max_epochs

        # Train the model for max_epochs epochs
        for epoch in range(max_epochs):
            if epoch / (max_epochs/10) == 0:
                print(f"Epoch {epoch} out of {max_epochs} running.")
            self.model.train(tagged_data,
                        total_examples=self.model.corpus_count,
                        epochs=self.model.epochs)

            # decrease the learning rate
            self.model.alpha -= learning_decay

        # reset learning rate after training
        self.model.alpha = prior_learning_rate

        # reduce memory usage
        self.model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    
    def find_similar_songs_known_song(self, song_id, n):

        '''
        Find and return the song IDs of the n most similar songs for a document the model was trained on

        :param song_id: Song ID for the song the most similar songs should be returned (int)
        :param n: Number indicating the number of most similar songs (int)

        :return: List of the most similar Song IDs (int)
        '''
        if song_id not in self.contained_doc_ids:
            # If the song_id is not trained on --> return random sample of doc IDs
            # TODO: Replace with getting the lyrics from the database and then predict
            print("Random")
            return random.sample(list(self.contained_doc_ids), n)
        else:
            # Find and return the most similar song IDs
            similar_docs = self.model.docvecs.most_similar(str(song_id), topn = n)
            return [int(similar_doc[0]) for similar_doc in similar_docs]

    def find_similar_songs_unknown_song(self, tokenized_lyric_list, n):
        '''
        Find and return the song IDs of the n most similar songs for a document the model was not trained on

        :param tokenized_lyric_list: List of preprocessed, tokenized lyrics (str)
        :param n: Number indicating the number of most similar songs (int)

        :return: List of the most similar Song IDs (int)
        '''

        # Convert the lyrics into a vector
        vector = self.model.infer_vector(tokenized_lyric_list)

        # Find and return the most similar song IDs
        similar_docs = self.model.docvecs.most_similar([vector], topn = n)
        print(similar_docs)
        return [int(similar_doc[0]) for similar_doc in similar_docs]
    


    def save_model(self, filepath_model, filepath_dict):
        '''
        Saves the current model and the contained_doc_ids

        :param filepath_model: path to the save location of model - File has to be of type .model  (str)
        :param filepath_dict: path to the save location of dictionary - File has to be of type .pkl  (str)
        '''
        self.model.save(filepath_model)

        with open(filepath_dict, 'wb') as file:
            dill.dump(self.contained_doc_ids, file)

    def load_model(self, filepath_model, filepath_dict):
        '''
        Loades a saved model and the contained_doc_ids

        :param filepath_model: path to the saved location of model - File has to be of type .model  (str)
        :param filepath_dict: path to the saved location of dictionary - File has to be of type .pkl  (str)
        '''
        self.model = Doc2Vec.load(filepath_model)

        with open(filepath_dict, 'rb') as file:
            self.contained_doc_ids = dill.load(file)

'''

# Set path as needed for Preprocessor class
path = os.path.abspath(__file__)
dname = os.path.dirname(os.path.dirname(path))
os.chdir(dname)


print(os.getcwd())
import sys
sys.path.append(os.getcwd())

from helpers.misc import load_yaml
from ETL.preprocessing import Preprocessor
config = load_yaml("config/config.yaml")
preprocessor = Preprocessor(config)


# Train on current data
import pandas as pd
data = pd.read_csv(os.path.join("features", "data-song_v1.csv"))

begin = time.time()
preprocessed_lyrics = [None] * data.shape[0]
for idx, row in data.iterrows():
    preprocessed_lyrics[idx] = preprocessor.preprocess(row["SongLyrics"])

ids = list(data["SongID"])

print(f"Data preprocessed in {begin - time.time()}")
begin = time.time()

rec_eng = RecommendationEngine()
rec_eng.load_model("word2vec2.model", "rec_model.pkl")

begin = time.time()
print(rec_eng.find_similar_songs_known_song(724643, 10))
print(f"Retrieval took: {time.time() - begin}")
#print(rec_eng.find_similar_songs_known_song(20, 10))


#rec_eng.train(preprocessed_lyrics, ids, max_epochs = 50)

#print(f"Engine trained in {begin - time.time()}")



#strg = preprocessor.preprocess("I've forgotten all the rest")
#print(rec_eng.find_similar_songs_unknown_song(strg, 3))
#begin = time.time()
#rec_eng.save_model("word2vec2.model", "rec_model.pkl")
#print(f"Model saved in {begin - time.time()}")

#rec_eng.load_model("word2vec2.model", "rec_model.pkl")
#print(rec_eng.contained_doc_ids)
#print(rec_eng.find_similar_songs_known_song(3, 1))




data = [["i","love","machine","learning.","its","awesome."],["i","love","coding","in","python"],["love","building","chatbots"],["chat","amagingly","well"]]
ids = [1,2,3,4]

rec_eng = RecommendationEngine()
rec_eng.train(data, ids, max_epochs = 10)

print(rec_eng.find_similar_songs_known_song(1, 1))
print(rec_eng.find_similar_songs_unknown_song(["i", "love", "chatbots"], 1))

rec_eng.save_model("word2vec2.model")
rec_eng.load_model("word2vec2.model")
print(rec_eng.find_similar_songs_known_song(1, 1))'''