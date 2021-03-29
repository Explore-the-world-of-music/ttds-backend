import logging
import os
import pickle
from datetime import datetime
import re
import collections
from numpy.lib.utils import _split_line

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from werkzeug.middleware.profiler import ProfilerMiddleware

from ETL.preprocessing import Preprocessor
from features.ngram_model import Query_Completer
from features.word_completion import Word_Completer
from helpers.misc import load_queries, load_yaml
from search_engine.indexer import Indexer
from search_engine.retrieval import execute_queries_and_save_results
from search_engine.system_evaluation import (
    calculate_average_precision, calculate_discounted_cumulative_gain,
    calculate_precision, calculate_recall, get_true_positives)

app = Flask(__name__)
CORS(app)

# if enabled, outputs all sql queries to the console
#app.config["SQLALCHEMY_ECHO"] = True 
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

from models.ArtistModel import ArtistModel
# Import the models after the database is initialised
from models.SongModel import SongModel

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
qc.load_model("./features/qc_model.pkl", "./features/qc_map_to_int.pkl",  "./features/qc_map_to_token.pkl")

wc = Word_Completer()
wc.load_model("./features/wc_model.pkl")

genres = db.session.query(SongModel.genre).with_entities(SongModel.genre).group_by(SongModel.genre).all()
languages = db.session.query(SongModel.language).with_entities(SongModel.language).group_by(SongModel.language).all()


logging.info("Ready")

@app.route("/")
def handle_root():
    return "<h1>The server is working! Try making an api call:</h1><a href=\"/api/songs/search?query=Never gonna give you up\">/api/songs/search?query=Never gonna give you up</a>"

@app.route("/api/songs/get_genres")
def handle_genres():
    return {"response": [result.genre for result in genres if result[0] is not None]}

@app.route("/api/songs/get_languages")
def handle_languages():
    return {"response": [result.language for result in languages if result[0] is not None]}

@app.route("/api/songs/search")
def handle_songs():
    """
    Returns a list of relevant songs
    :param query: query text (str)
    :param years: year range (list of int)
    :param artists: artists (list of str)
    :param genre: genres (list of str)
    :return: results (json)
    """
    query = request.args.get("query", "")
    years = request.args.get("years", "1960, 2021")
    artists = request.args.get("artists", "")
    genres = request.args.get("genres", "")
    language = request.args.get("language", "")

    if config["retrieval"]["perform_system_evaluation"]:
        print("Perform system evaluation")
        # Load and run queries
        queries_num, queries = load_queries('system_evaluation/queries.system_evaluation.txt')
        results_data_frame = pd.DataFrame()
        for query_num, query in zip(queries_num, queries):
            _, results_data_frame_tmp, _ = execute_queries_and_save_results(query, indexer=indexer,preprocessor=preprocessor,
                                                                         config=config, SongModel=SongModel,
                                                                         ArtistModel = ArtistModel,
                                                                         query_num=query_num, rel_docs = set(range(1, indexer.total_num_docs+1)))
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

    query_list = []
    if years !="":
        years = years.split(",")
        if int(years[0]) != 1960 or int(years[1]) != 2021:
            query_list.append(SongModel.released.between(int(years[0]), int(years[1])))
    
    if artists != "":
        artists = artists.split(",")
        query_list.append(ArtistModel.name.in_(artists))
    
    if genres != "":
        genres = genres.split(",")
        query_list.append(SongModel.genre.in_(genres))
    
    if language != "":
        language = language.split(",")
        query_list.append(SongModel.language.in_(language))
    
    logging.info("Sending a query to the DB with advanced options filter")
    if len(query_list) > 0:
        filtered_songs = SongModel.query.with_entities(SongModel.id).join(ArtistModel).filter(*query_list).all()
        filtered_songs = set(song for (song,) in filtered_songs)
    else:
        filtered_songs = set(range(1, indexer.total_num_docs+1))
    logging.info("Recevided results from the DB with advanced options filter")
    
    # Perform search to be shown in front end
    logging.info("Starting index search")
    db_results, _ = execute_queries_and_save_results(query, indexer=indexer,preprocessor=preprocessor,
                                                     config=config, SongModel=SongModel, ArtistModel = ArtistModel, rel_songs= filtered_songs)
    logging.info("Index search complete")
    if len(db_results) == 0 and query[0] == "\"" and query[-1] == "\"" and len(query) > 2 and "\"" not in query[1:-1]:
        db_results, _ = execute_queries_and_save_results(query[1:-1], indexer=indexer,preprocessor=preprocessor,
                                                     config=config, SongModel=SongModel, ArtistModel = ArtistModel, rel_songs= filtered_songs)
    if len(db_results) == 0:
        return {"songs": []}

    result_dict = {id: score for id, score in db_results} # converting tuples into a dictionary
    logging.info("Getting relevant songs from DB")
    songs = SongModel.query.join(ArtistModel).filter(SongModel.id.in_(result_dict.keys())).all()
    logging.info("Done")

    results = [
        {
            "id": song.id,
            "name": song.name,
            "artist": song.artist.name,
            "lyrics": song.lyrics,
            "album": song.album,
            "image": song.album_image if song.album_image != None else song.artist.image,
            "rating": song.rating,
            "released": song.released,
            "genre": song.genre
        } for song in songs]
    query_set = set(preprocessor.preprocess(re.sub('|,\)\(&-"#', '', query).replace("*", "")))

    if "" in query_set:
        query_set.remove("")
    extras = set()

    for term in query_set:
        extras.update(preprocessor.preprocess(preprocessor.replace_replacement_patterns(term)))
    query_set.update(extras)

    for song in results:
        line_matches = []
        best_line = 0
        best_line_value = 0
        new_lyrics = []
        song["lyrics"] = song["lyrics"].replace("\\n", "\n")
        for lyric_line in song["lyrics"].split("\n"):
            line_matches.append(0)
            if lyric_line != "":
                for word in lyric_line.split(" "):
                    if len(preprocessor.preprocess(word)) > 0 and ((preprocessor.preprocess(word)[0] in query_set) or set(preprocessor.preprocess(preprocessor.replace_replacement_patterns(word))).issubset(query_set)):
                        new_lyrics.append(f"<b>{word}</b>")
                        line_matches[-1] += 1
                    else:
                        new_lyrics.append(word)
                if line_matches[-1] > best_line_value:
                    best_line = len(line_matches)-1
                    best_line_value = line_matches[-1]
            new_lyrics.append("\n")
        new_lyrics = " ".join(new_lyrics)

        split_lyrics = new_lyrics.split("\n")
        limit = 9
        if "" in split_lyrics[best_line:best_line+limit] and 4 <= split_lyrics[best_line:best_line+limit].index("") <= 10:
            song["lyrics"] = "\n".join(split_lyrics[best_line:best_line + split_lyrics[best_line:best_line+limit].index("") + 1])
        else:
            song["lyrics"] = "\n".join(split_lyrics[best_line:best_line+limit])

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
    
    artists = ArtistModel.query.with_entities(ArtistModel.name).filter(ArtistModel.name.ilike(f"%{query}%")).limit(50).all()
    results = [
        {
            "artist": artist[0],
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

    if "&" in query or "|" in query or "#" in query or "\"" in query:
        return {"suggestions": []} 

    # if ends with a space predict the next word, otherwise predict the rest of the current word
    if query[-1] == " ":
        results = qc.predict_next_token(query[:-1])
    else:
        results = wc.predict_token(query, 5)
    if results == None:
        results = []
    return {"suggestions": results}


@app.route("/api/songs/get_lyrics")
# http://127.0.0.1:5000/api/songs/get_lyrics?id=1
def handle_lyrics():
    """
    Returns song lyrics and metadata
    :param query: song id (int)
    :return: results (json)
    """
    id = request.args.get("id", "")

    result = SongModel.query.join(ArtistModel).filter(SongModel.id == id).scalar()
    
    results = {
            "id": result.id,
            "name": result.name,
            "artist": result.artist.name,
            "lyrics": result.lyrics,
            "album": result.album,
            "image": result.album_image if result.album_image != None else result.artist.image,
            "rating": result.rating,
            "released": result.released,
            "genre": result.genre,
            "bpm": result.bpm,
            "key": result.key,
            "topic_id": result.topic_id,
            "length": result.length,
            "language": result.language
        } 

    recom_id = [result.rec1, result.rec2, 
                result.rec3, result.rec4,
                result.rec5]

    recom_songs = SongModel.query.join(ArtistModel).filter(SongModel.id.in_(recom_id)).all()
    recom_list = [{ "id": r.id,
        "name": r.name,
        "artist": r.artist.name,
        "album": r.album,
        "image": r.album_image if r.album_image != None else r.artist.image,
    } for r in recom_songs]

    results["recommendations"] = recom_list

    return results