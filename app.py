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

app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/test')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Import the model after the database is initialised
from models.SongModel import SongModel

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

@app.route('/api/songs')
def handle_songs():
    """
    Returns a list of relevant songs
    :param query: query text (str)
    :return: results (json)
    """
    query = request.args.get('query')

    db_results = execute_queries_and_save_results(query, search_type="boolean", indexer=indexer,
                                                       preprocessor=preprocessor, config=config)
    songs = SongModel.query.filter(SongModel.id.in_(db_results)).all()
    print(songs)
    results = [
        {
            "name": song.name,
            "author": song.author,
            "lyrics": song.lyrics
        } for song in songs]
    return {"count": len(results), "songs": results}
