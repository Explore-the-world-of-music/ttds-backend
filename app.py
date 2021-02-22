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
        app.config['SQLALCHEMY_DATABASE_URI'] = passfile.readline()
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Import the models after the database is initialised
from models.SongModel import SongModel
from models.ArtistModel import ArtistModel

# Stop time
# Full run: 23 seconds
# Run without creating but only loading index: 0-1 seconds
dt_string_START = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
logging.warning(f'START date and time = {dt_string_START}')

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


@app.route('/')
def handle_root():
    return '<h1>The server is working! Try making an api call:</h1><a href=\"/api/songs?query=Never gonna give you up\">/api/songs?query=Never gonna give you up</a>'


@app.route('/api/songs')
def handle_songs():
    """
    Returns a list of relevant songs
    :param query: query text (str)
    :return: results (json)
    """
    query = request.args.get('query')

    db_results = execute_queries_and_save_results(query, search_type="boolean_and_tfidf", indexer=indexer,
                                                  preprocessor=preprocessor, config=config)
    songs = SongModel.query.join(ArtistModel).filter(
        SongModel.id.in_(db_results)).all()

    results = [
        {
            "id": song.id,
            "name": song.name,
            "artist": song.artist.name,
            "lyrics": song.lyrics.replace("\\n", "\n"),
            "album": song.album,
            "image": song.artist.image,
            "rating": song.rating,
            "released": song.released,
            "genre": song.genre
        } for song in songs]
    return {"songs": results}
