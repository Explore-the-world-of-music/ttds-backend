from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

import os
import logging
from datetime import datetime

from ETL.preprocessing import Preprocessor
from helpers.misc import load_yaml
from search_engine.indexer import Indexer
from search_engine.retrieval import execute_queries_and_save_results

app = Flask(__name__)
CORS(app)

if os.path.isfile(".password") and os.access(".password", os.R_OK):
    with open(".password", "r") as passfile:
        app.config["SQLALCHEMY_DATABASE_URI"] = passfile.readline()
else:
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL")

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Import the models after the database is initialised
from models.SongModel import SongModel
from models.ArtistModel import ArtistModel

# Stop time
# Full run: 23 seconds
# Run without creating but only loading index: 0-1 seconds
dt_string_START = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
logging.warning(f"START date and time = {dt_string_START}")

# Load config
config = load_yaml("config/config.yaml")

# Initialize preprocessor instance
preprocessor = Preprocessor(config)

# Load data
doc_ids, raw_doc_texts = preprocessor.load_data_from_db(SongModel)

# Initiate indexer instance
indexer = Indexer(config)

# Build index
indexer.build_index(preprocessor, doc_ids, raw_doc_texts)

# Save index
indexer.store_index()

# Add doc ids as index attribute
indexer.add_all_doc_ids(doc_ids)

# Load index (for testing)
indexer.index = indexer.load_index()

@app.route("/")

def handle_root():
    return "<h1>The server is working! Try making an api call:</h1><a href=\"/api/songs/search?query=Never gonna give you up\">/api/songs/search?query=Never gonna give you up</a>"

@app.route("/api/songs/get_genres")
def handle_genres():
    results = db.session.query(SongModel.genre).group_by(SongModel.genre).all()
    return {"genres": [result.genre for result in results if result.genre is not None]}

@app.route("/api/songs/search")
def handle_songs():
    """
    Returns a list of relevant songs
    :param query: query text (str)
    :param years: year range (list of int)
    :param artist: artist (list of str)
    :param genre: genres (list of str)
    :return: results (json)
    """
    query = request.args.get("query", "")
    years = request.args.get("years", "1960, 2021")
    artists = request.args.get("artists", "")
    genres = request.args.get("genres", "")

    db_results = execute_queries_and_save_results(query, search_type="boolean_and_tfidf", indexer=indexer,
                                                  preprocessor=preprocessor, config=config)

    if db_results == None:
        return {"songs": []}

    result_dict = {id: score for id, score in db_results} # converting tuples into a dictionary

    query_list = [SongModel.id.in_(result_dict.keys())]
    if years !="":
        years = years.split(",")
        query_list.append(SongModel.released.between(int(years[0]), int(years[1])) )
    
    if artists != "":
        artists = artists.split(",")
        query_list.append(SongModel.artist._in(artists))
    
    if genres != "":
        genres = genres.split(",")
        query_list.append(SongModel.genre._in(genres))
    
    songs = SongModel.query.join(ArtistModel).filter(*query_list).all()

    results = [
        {
            "id": song.id,
            "name": song.name,
            "artist": song.artist.name,
            "lyrics": song.lyrics,
            "album": song.album,
            "image": song.artist.image,
            "rating": song.rating,
            "released": song.released,
            "genre": song.genre
        } for song in songs]

    # Extra preporcessing before lyrics are returned:
    # 1. Replace all \\n with \n
    # 2. If there is a split between 4 and 10 lines, use that
    # 3. Otherwise, just return the first 8 lines
    # TODO: add different method for phrase search.
    for song in results:
        song["lyrics"] = song["lyrics"].replace("\\n", "\n")
        split_lyrics = song["lyrics"].split("\n")
        if "" in split_lyrics and 4 <= split_lyrics.index("") <= 10:
            song["lyrics"] = "\n".join(split_lyrics[:split_lyrics.index("")])
        else:
            song["lyrics"] = "\n".join(split_lyrics[:8])

    # sort results based on their score
    results.sort(key= lambda x: result_dict[x["id"]], reverse=True)

    return {"songs": results}

# http://127.0.0.1:5000/api/artist/get_artist?query=enslav
@app.route('/api/artist/get_artist')
def handle_artists():
    """
    Returns a list of suggested/relevant artist names 
    :param artist name (str)
    :return results (json)
    """
    query = request.args.get('query').lower()
    
    artists = ArtistModel.query.all()
    results = [
        {
            "id": artist.id,
            "artist": artist.name,
        } for artist in artists if query in artist.name.lower()]
    return {"artists": results}
