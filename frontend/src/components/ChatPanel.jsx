export function ChatPanel({
  messages,
  boostEnabled,
  onBoostChange,
  onSend,
  sending,
  mode,
  onModeChange,
}) {
  function handleSubmit(e) {
    e.preventDefault()
    const fd = new FormData(e.target)
    const text = (fd.get('message') || '').trim()
    if (!text) return
    onSend(text)
    e.target.reset()
  }

  return (
    <section className="chat-panel">
      <div className="chat-toolbar">
        <label>
          Mode{' '}
          <select
            value={mode}
            onChange={(e) => onModeChange(e.target.value)}
            disabled={sending}
          >
            <option value="chat">Chat</option>
            <option value="quiz">Quiz</option>
            <option value="compare">Compare</option>
            <option value="summary">Summary</option>
          </select>
        </label>
        <label className="boost-toggle">
          <input
            type="checkbox"
            checked={boostEnabled}
            onChange={(e) => onBoostChange(e.target.checked)}
          />
          Boosted explanation
        </label>
      </div>

      <div className="messages">
        {messages.length === 0 ? (
          <p className="muted">Ask a question about LING 487 course material.</p>
        ) : (
          messages.map((m) => <MessageBlock key={m.id} m={m} />)
        )}
      </div>

      <form className="composer" onSubmit={handleSubmit}>
        <input name="message" placeholder="Message…" autoComplete="off" />
        <button type="submit" disabled={sending}>
          Send
        </button>
      </form>
    </section>
  )
}

function MessageBlock({ m }) {
  if (m.role === 'user') {
    return (
      <div className="msg user">
        <div className="msg-body">{m.content}</div>
      </div>
    )
  }
  return (
    <div className="msg assistant">
      <div className="msg-block course">
        <div className="label">Course Answer</div>
        <div className="msg-body">{m.course_answer || ''}</div>
      </div>
      {m.boosted_explanation ? (
        <div className="msg-block boost">
          <div className="label">Boosted Explanation</div>
          <div className="msg-body">{m.boosted_explanation}</div>
        </div>
      ) : null}
    </div>
  )
}
