from app.extensions import db


class LectureChunk(db.Model):
    """
    Indexed lecture material for retrieval (v1: keyword-based; embeddings later).

    SQLite table ``lecture_chunks`` (``flask --app wsgi init-db``):

    .. code-block:: sql

        CREATE TABLE lecture_chunks (
            id INTEGER NOT NULL PRIMARY KEY,
            lecture_number INTEGER NOT NULL,
            topic VARCHAR(512) NOT NULL,
            keywords TEXT NOT NULL,
            source_excerpt TEXT NOT NULL,
            clean_explanation TEXT NOT NULL,
            sample_questions TEXT,
            sample_answer TEXT
        );
        CREATE INDEX ix_lecture_chunks_lecture_number ON lecture_chunks (lecture_number);
        CREATE INDEX ix_lecture_chunks_topic ON lecture_chunks (topic);

    Seed JSON may use ``source_text`` or ``content``; both map to ``source_excerpt``.
    ``keywords`` is stored as JSON array text (e.g. ``["neural", "weights"]``).
    """

    __tablename__ = "lecture_chunks"

    id = db.Column(db.Integer, primary_key=True)
    lecture_number = db.Column(db.Integer, nullable=False, index=True)
    topic = db.Column(db.String(512), nullable=False, index=True)
    keywords = db.Column(db.Text, nullable=False)
    source_excerpt = db.Column(db.Text, nullable=False)
    clean_explanation = db.Column(db.Text, nullable=False)
    sample_questions = db.Column(db.Text, nullable=True)
    sample_answer = db.Column(db.Text, nullable=True)
