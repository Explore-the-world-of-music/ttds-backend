from app import db
#from sqlalchemy.dialects.postgresql import JSON

class ArtistModel(db.Model):
    __tablename__ = 'artists'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String())
    image = db.Column(db.String())
    rating = db.Column(db.SmallInteger())

    def __repr__(self):
        return f"<Artist {self.name}>"
