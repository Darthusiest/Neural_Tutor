import { Link } from 'react-router-dom'

export function Sidebar({ sessions, activeId, onNewChat, disabled }) {
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
          <li key={s.id}>
            <Link
              className={s.id === activeId ? 'session-link active' : 'session-link'}
              to={`/chat/${s.id}`}
            >
              <span className="session-title">{s.title}</span>
              <span className="session-meta">{s.mode}</span>
            </Link>
          </li>
        ))}
      </ul>
    </aside>
  )
}
