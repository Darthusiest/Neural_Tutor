from app.extensions import db


class ChatSession(db.Model):
    __tablename__ = "chat_sessions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    title = db.Column(db.String(512), nullable=False, default="New chat")
    mode = db.Column(
        db.String(32), nullable=False, default="chat"
    )  # chat | quiz | compare | summary
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    updated_at = db.Column(
        db.DateTime, server_default=db.func.now(), onupdate=db.func.now()
    )

    user = db.relationship("User", back_populates="chat_sessions")
    messages = db.relationship(
        "Message", back_populates="session", cascade="all, delete-orphan"
    )


class Message(db.Model):
    __tablename__ = "messages"

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(
        db.Integer, db.ForeignKey("chat_sessions.id"), nullable=False, index=True
    )
    role = db.Column(db.String(32), nullable=False)  # user | assistant | system
    content_text = db.Column(db.Text, nullable=True)
    payload_json = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

    session = db.relationship("ChatSession", back_populates="messages")
    response_variant = db.relationship(
        "ResponseVariant", back_populates="message", uselist=False
    )
    retrieval_logs = db.relationship("RetrievalLog", back_populates="message")
    feedback = db.relationship("Feedback", back_populates="message", uselist=False)
