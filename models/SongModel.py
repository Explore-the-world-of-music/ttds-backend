from app import db
#from sqlalchemy.dialects.postgresql import JSON

class SongModel(db.Model):
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
