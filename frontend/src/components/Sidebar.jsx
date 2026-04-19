import { Link } from 'react-router-dom'

export function Sidebar({
  sessions,
  activeId,
  onNewChat,
  onRenameSession,
  onDeleteSession,
  disabled,
}) {
  return (
    <aside className="sidebar">
      <button
        type="button"
        className="new-chat-btn"
        onClick={onNewChat}
        disabled={disabled}
      >
        New chat
      </button>
      <ul className="session-list">
        {sessions.map((s) => (
          <li key={s.id} className="session-row">
            <Link
              className={s.id === activeId ? 'session-link active' : 'session-link'}
              to={`/chat/${s.id}`}
            >
              <span className="session-title">{s.title}</span>
              <span className="session-meta">{s.mode}</span>
            </Link>
            <button
              type="button"
              className="session-rename"
              onClick={(e) => {
                e.preventDefault()
                onRenameSession(s.id, s.title)
              }}
              disabled={disabled}
              aria-label={`Rename chat ${s.title}`}
              title="Rename chat"
            >
              ✎
            </button>
            <button
              type="button"
              className="session-delete"
              onClick={(e) => {
                e.preventDefault()
                onDeleteSession(s.id)
              }}
              disabled={disabled}
              aria-label={`Delete chat ${s.title}`}
              title="Delete chat"
            >
              ×
            </button>
          </li>
        ))}
      </ul>
    </aside>
  )
}
