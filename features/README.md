# Feature Documentation
## Query Completion (ngram_model.py)
This module trains and stores an ngram model to predict the next word from an inputted query. It returns the whole sentence with the five most probable continuations. The pretrained model and the relevant mappings for our data is available for download [here](https://www.dropbox.com/sh/hmpphonwiyxyc0q/AADCzZIz2Aa6I93NPQycLlQoa?dl=0) and can be loaded using the `load_model` function.

The class `Query_Completer` needs the parameter for `n` for initialization (n=3 in the current case).

### Important Functions for Connection
- `add_single_lyric(lyrics)` adds a new unprocessed lyric (str) to the model
- `save_model(model_filepath, map_to_int_filepath, map_to_token_filepath)` saves the models in pickle files - default values exist
- `load_model(model_filepath, map_to_int_filepath, map_to_token_filepath)` restores the models from pickle files - default values exist
- `predict_next_token(current_query)` predicts the five most probable continuations for a unprocessed query based on the last n-1 tokens of the query
- `reduce_model(cutoff)` reduces the model by removing the identifiers which occured less than 'cutoff' times

### Examples
```Py
data = pd.read_csv("data-song-sample.csv") # Load the provided data

# Instanciate the model
qc = Query_Completer(n = 3)

# Add all Lyrics to the model
for idx, row in data.iterrows():
    qc.add_single_lyric(row["SongLyrics"])

# Reducing the size of the model    
qc.reduce_model(5)

# Saving and reloading of the model
qc.save_model()
qc.load_model()

# Predicting some continuations
print(qc.predict_next_token("deux trois"))
# Output: ['deux trois quatre', 'deux trois mousseau', 'deux trois no', 'deux trois calibre', 'deux trois mots']
print(qc.predict_next_token("Oops I"))
# Output: ['Oops I did', 'Oops I mean', 'Oops I meant', 'Oops I got', 'Oops I forgot']
print(qc.predict_next_token("Es ragen aus ihrem aufgeschlitzten Bauch"))
# Output: ['Es ragen aus ihrem aufgeschlitzten Bauch rippen'] <- "aufgeschlitzten Bauch" only occured in this context and wont be part of the model after reduction
```
---

## Word Completion (word_completion.py)
This module trains and stores an ngram model to predict the next word from an inputted query. It returns the whole query with the five most probable completed tokens. The pretrained model is available for download [here](https://www.dropbox.com/s/bgifg45sbe3jbgl/wc_model.pkl?dl=0) and can be loaded using the `load_model` function.

### Important Functions for Connection
- `add_single_lyric(lyrics)` adds a new unprocessed lyric (str) to the model
- `save_model(model_filepath = "wc_model.pkl")` saves the models in a pickle file
- `load_model(model_filepath = "wc_model.pkl")` restores the models from a pickle file
- `predict_token(query, n)` predicts the `n` most probable words for the last splittable token

### Examples
```Py
data = pd.read_csv("data-song-sample.csv") # Load the provided data

# Instanciate the model
wc = Word_Completer()

# Add all Lyrics to the model
for idx, row in data.iterrows():
    wc.add_single_lyric(row["SongLyrics"])

# Saving and reloading of the model
qc.save_model()
qc.load_model()

# Predicting
n = 5
print(wc.predict_token("Hell",n))
# Output: ['hell', 'hello', 'hella', 'hells', 'hellish']
print(wc.predict_token("Oop",n))
# Output: [' oops', ' oop', ' oopsie', ' oopsy', ' oopiri']
print(wc.predict_token("World qua",n)) # In this case, only the last word is completed and appended to the former words
# Output: ['world quand', 'world quando', 'world quarter', 'world quanto', 'world qua']

```

---

## Recommendation Module (recommender.py)
This module trains and staores a Doc2Vec model to predict the most similar lyrics of a given song. The class `RecommendationEngine` contains the model and a dictionary of the contained Song IDs. The trained [model](https://www.dropbox.com/sh/p9kpfo843mftoz6/AABBaIezWkNlNshhOx6OyZkNa?dl=0) is available for download .

### Important Functions for Connection
- `save_model(filepath_model, filepath_dict)` saves the model in a .model and the Song ID dictionary in a pickle file
- `load_model(filepath_model, filepath_dict)` restores the model from a .model and the Song ID dictionary from a pickle file
- `find_similar_songs_known_song(song_id, n)` return the Song IDs for the `n` most similar songs for the given song ID


```Py
rec_eng = RecommendationEngine()
rec_eng.load_model("word2vec2.model", "rec_model.pkl")

print(rec_eng.find_similar_songs_known_song(168861, 10))
# Output: [137760, 137761, 461554, 498408, 479446, 157127, 438013, 482450, 285817, 254288]
```

