from app import db
#from sqlalchemy.dialects.postgresql import JSON
from models.ArtistModel import ArtistModel


class SongModel(db.Model):
    __tablename__ = 'songs'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String())
    artist_id = db.Column(db.Integer(), db.ForeignKey("artists.id"))
    lyrics = db.Column(db.Text())
    album = db.Column(db.String())
    released = db.Column(db.Integer())
    rating = db.Column(db.SmallInteger())
    genre = db.Column(db.String())
    language = db.Column(db.String())
    length = db.Column(db.Integer())
    bpm = db.Column(db.Integer())
    key = db.Column(db.Integer())
    topic_id = db.Column(db.Integer())
    rec1 = db.Column(db.Integer())
    rec2 = db.Column(db.Integer())
    rec3 = db.Column(db.Integer())
    rec4 = db.Column(db.Integer())
    rec5 = db.Column(db.Integer())
    
    artist = db.relationship("ArtistModel")

    def __repr__(self):
        return f"<Song {self.name}>"
