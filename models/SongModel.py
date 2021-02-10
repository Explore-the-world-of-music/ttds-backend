from app import db
#from sqlalchemy.dialects.postgresql import JSON

class SongModel(db.Model):
    __tablename__ = 'songs'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String())
    artist = db.Column(db.String())
    lyrics = db.Column(db.Text())
    album = db.Column(db.String())
    image = db.Column(db.String())
    released = db.Column(db.SmallInteger())
    genre = db.Column(db.String())

    def __init__(self, name, lyrics, artist, album, image, released, genre):
        self.name = name
        self.artist = artist
        self.lyrics = lyrics
        self.album = album
        self.image = image
        self.released = released
        self.genre = genre

    def __repr__(self):
        return f"<Song {self.name}>"
