from app.extensions import db


class LectureChunk(db.Model):
    """Indexed lecture material for retrieval (v1: keyword-based)."""

    __tablename__ = "lecture_chunks"

    id = db.Column(db.Integer, primary_key=True)
    topic = db.Column(db.String(512), nullable=False, index=True)
    lecture_number = db.Column(db.Integer, nullable=False, index=True)
    keywords = db.Column(db.Text, nullable=False)
    explanation = db.Column(db.Text, nullable=False)
    example_qa = db.Column(db.Text, nullable=True)
