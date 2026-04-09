from app.extensions import db


class LectureChunk(db.Model):
    """Indexed lecture material for retrieval (v1: keyword-based; embeddings later)."""

    __tablename__ = "lecture_chunks"

    id = db.Column(db.Integer, primary_key=True)
    lecture_number = db.Column(db.Integer, nullable=False, index=True)
    topic = db.Column(db.String(512), nullable=False, index=True)
    keywords = db.Column(db.Text, nullable=False)
    source_excerpt = db.Column(db.Text, nullable=False)
    clean_explanation = db.Column(db.Text, nullable=False)
    sample_questions = db.Column(db.Text, nullable=True)
    sample_answer = db.Column(db.Text, nullable=True)
