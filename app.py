from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.middleware.profiler import ProfilerMiddleware


import os
import logging
import pickle
from datetime import datetime
import pandas as pd

from ETL.preprocessing import Preprocessor
from features.ngram_model import Query_Completer
from features.word_completion import Word_Completer
from helpers.misc import load_yaml, load_queries
from search_engine.indexer import Indexer
from search_engine.retrieval import execute_queries_and_save_results
from search_engine.system_evaluation import get_true_positives, calculate_precision, calculate_recall, \
    calculate_average_precision, calculate_discounted_cumulative_gain

app = Flask(__name__)
CORS(app)

# if enabled, outputs all sql queries to the console
app.config["SQLALCHEMY_ECHO"] = True 
# uncomment these 2 lines to enable profiling
# app.config['PROFILE'] = True
# app.wsgi_app = ProfilerMiddleware(app.wsgi_app, restrictions=[30])

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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

logging.info("Start")

# Load config
config = load_yaml("config/config.yaml")

# Initialize preprocessor instance
preprocessor = Preprocessor(config)

# Load data
# doc_ids, raw_doc_data = preprocessor.load_data_from_db(SongModel, ArtistModel)

# with open("data.pickle", "rb") as song_file:
#     doc_ids, raw_doc_data = pickle.load(song_file)

# logging.info("Finished loading data")

# Initiate indexer instance
indexer = Indexer(config)

# Build index
# indexer.build_index(preprocessor, doc_ids, raw_doc_data)

# logging.info("Finished building index")

# Save index
# indexer.store_index()

# logging.info("Index saved")

# Add doc ids as index attribute
# indexer.add_all_doc_ids(doc_ids)
total_num_docs = SongModel.query.with_entities(SongModel.id).count()
# Load index (for testing)
indexer.index = indexer.load_index(total_num_docs, False)
qc = Query_Completer(n = 3)
#qc.load_model("./features/qc_model.pkl", "./features/qc_map_to_int.pkl",  "./features/qc_map_to_token.pkl")

wc = Word_Completer()
#wc.load_model("./features/wc_model.pkl")

logging.info("Ready")

@app.route("/")
def handle_root():
    return "<h1>The server is working! Try making an api call:</h1><a href=\"/api/songs/search?query=Never gonna give you up\">/api/songs/search?query=Never gonna give you up</a>"

@app.route("/api/songs/get_genres")
def handle_genres():
    results = db.session.query(SongModel.genre).with_entities(SongModel.genre).group_by(SongModel.genre).all()
    return {"genres": [result.genre for result in results if result[0] is not None]}

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

    if config["retrieval"]["perform_system_evaluation"]:
        print("Perform system evaluation")
        # Load and run queries
        queries_num, queries = load_queries('system_evaluation/queries.system_evaluation.txt')
        results_data_frame = pd.DataFrame()
        for query_num, query in zip(queries_num, queries):
            _, results_data_frame_tmp = execute_queries_and_save_results(query, indexer=indexer,preprocessor=preprocessor,
                                                                         config=config, SongModel=SongModel,
                                                                         ArtistModel = ArtistModel,
                                                                         query_num=query_num)
            results_data_frame = results_data_frame.append(results_data_frame_tmp)

        # Todo: Take out again once we have real values
        dummy_correct_results = results_data_frame[["query_number", "doc_number", "score"]].reset_index(drop=True)
        dummy_correct_results["relevance"] = [round(x, 0) for x in dummy_correct_results["score"]]
        dummy_correct_results.drop(columns=["score"], inplace=True)
        dummy_correct_results.to_csv("system_evaluation/correct_search_results.csv", index=False)

        # Load correct results
        df_correct_search_results = pd.read_csv("system_evaluation/correct_search_results.csv")
        df_correct_search_results["query_number"] = df_correct_search_results["query_number"].astype(str)

        # Calculate precision
        df_evaluation_results = calculate_precision(results_data_frame, df_correct_search_results,
                                                    cutoff=config["retrieval"]["number_ranked_documents"])

        # Calculate recall
        df_evaluation_results = pd.merge(df_evaluation_results,
                                         calculate_recall(results_data_frame, df_correct_search_results,
                                                          cutoff=config["retrieval"]["number_ranked_documents"]),
                                         how="left", on=["query_number"])

        # Calculate AP
        df_evaluation_results = pd.merge(df_evaluation_results,
                                         calculate_average_precision(results_data_frame, df_correct_search_results),
                                         how="left",on =["query_number"])

        # Calculate nDCG@10
        df_evaluation_results = pd.merge(
            df_evaluation_results, calculate_discounted_cumulative_gain(results_data_frame,df_correct_search_results,
                                                                        rank=config["retrieval"]["number_ranked_documents"]),
            how="left", on=["query_number"])

        # Output results in the appropriate format
        df_evaluation_results = df_evaluation_results.round(3)
        print(f' The evaluation results are:\n {df_evaluation_results}')
        df_evaluation_results.to_csv("system_evaluation/results_system_evaluation.csv", index=False)

    # Perform search to be shown in front end
    logging.info("Starting index search")
    db_results, _ = execute_queries_and_save_results(query, indexer=indexer,preprocessor=preprocessor,
                                                     config=config, SongModel=SongModel, ArtistModel = ArtistModel)
    logging.info("Index search complete")
    if db_results == None:
        return {"songs": []}

    result_dict = {id: score for id, score in db_results} # converting tuples into a dictionary

    query_list = [SongModel.id.in_(result_dict.keys())]
    if years !="":
        years = years.split(",")
        query_list.append(SongModel.released.between(int(years[0]), int(years[1])) )
    
    if artists != "":
        artists = artists.split(",")
        query_list.append(ArtistModel.id.in_(artists))
    
    if genres != "":
        genres = genres.split(",")
        query_list.append(SongModel.genre.in_(genres))
    
    logging.info("Sending a query to the DB")
    songs = SongModel.query.join(ArtistModel).filter(*query_list).all()
    logging.info("Recevided results from the DB")

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
        if not config["retrieval"]["result_checking"]:
            if "" in split_lyrics and 4 <= split_lyrics.index("") <= 10:
                song["lyrics"] = "\n".join(split_lyrics[:split_lyrics.index("")])
            else:
                song["lyrics"] = "\n".join(split_lyrics[:8])

    # sort results based on their score
    results.sort(key= lambda x: result_dict[x["id"]], reverse=True)

    return {"songs": results}

# http://127.0.0.1:5000/api/artists/get_artist?query=enslav
@app.route('/api/artists/get_artist')
def handle_artists():
    """
    Returns a list of suggested/relevant artist names 
    :param artist name (str)
    :return results (json)
    """
    query = request.args.get('query').lower()
    
    artists = ArtistModel.query.with_entities(ArtistModel.id, ArtistModel.name).filter(ArtistModel.name.ilike(f"%{query}%")).all()
    results = [
        {
            "id": artist[0],
            "artist": artist[1],
        } for artist in artists]
    return {"results": results}

@app.route('/api/songs/query_autocomplete')
def handle_autocomplete():
    """
    Returns a list of suggested queries
    :param query (str)
    :return results (json)
    """

    query = request.args.get('query')

    # if ends with a space predict the next word, otherwise predict the rest of the current word
    if query[-1] == " ":
        print("!", query[:-1], "!")
        results = qc.predict_next_token(query[:-1])
    else:
        results = wc.predict_token(query, 5)
    if results == None:
        results = []
    return {"suggestions": results}
