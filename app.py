import time
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os

#test

app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/test')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

class SongsModel(db.Model):
    __tablename__ = 'songs'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String())
    author = db.Column(db.String())
    lyrics = db.Column(db.Text())

    def __init__(self, name, lyrics, author):
        self.name = name
        self.author = author
        self.lyrics = lyrics

    def __repr__(self):
        return f"<Song {self.name}>"


@app.route('/api/time')
def get_current_time():
    return jsonify({'time': time.time()})


@app.route('/api/songs', methods=['POST', 'GET'])
def handle_songs():
    songs = SongsModel.query.all()
    results = [
        {
            "name": song.name,
            "author": song.author,
            "lyrics": song.lyrics
        } for song in songs]

    return {"count": len(results), "songs": results}

@app.route('/api/songs/add')
def handle_add_songs():
    new_song = SongsModel(name=request.args.get('name'), author=request.args.get('author'), lyrics=request.args.get('lyrics'))
    db.session.add(new_song)
    db.session.commit()
    return {"message": f"song {new_song.name} has been created successfully."}
