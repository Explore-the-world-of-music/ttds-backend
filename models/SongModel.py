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
    
    
    artist = db.relationship("ArtistModel")

    def __repr__(self):
        return f"<Song {self.name}>"
