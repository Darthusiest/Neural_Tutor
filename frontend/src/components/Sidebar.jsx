import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'

export function Sidebar({
  sessions,
  activeId,
  onNewChat,
  onRenameSession,
  onDeleteSession,
  disabled,
}) {
  const [editingId, setEditingId] = useState(null)
  const [renameDraft, setRenameDraft] = useState('')
  const renameInputRef = useRef(null)

  useEffect(() => {
    if (editingId != null) {
      const el = renameInputRef.current
      el?.focus()
      el?.select()
    }
  }, [editingId])

  function beginRename(s) {
    setEditingId(s.id)
    setRenameDraft(s.title || '')
  }

  function cancelRename() {
    setEditingId(null)
    setRenameDraft('')
  }

  async function commitRename(sid) {
    try {
      await onRenameSession(sid, renameDraft)
      setEditingId(null)
      setRenameDraft('')
    } catch {
      renameInputRef.current?.focus()
    }
  }

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
          <li
            key={s.id}
            className={
              editingId === s.id ? 'session-row session-row--editing' : 'session-row'
            }
          >
            {editingId === s.id ? (
              <form
                className="session-rename-form"
                onSubmit={(e) => {
                  e.preventDefault()
                  commitRename(s.id)
                }}
              >
                <input
                  ref={renameInputRef}
                  type="text"
                  name="title"
                  autoComplete="off"
                  maxLength={512}
                  value={renameDraft}
                  onChange={(e) => setRenameDraft(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Escape') {
                      e.preventDefault()
                      cancelRename()
                    }
                  }}
                  disabled={disabled}
                  aria-label="Chat name"
                />
                <div className="session-rename-bar">
                  <button type="submit" className="session-rename-save" disabled={disabled}>
                    Save
                  </button>
                  <button
                    type="button"
                    className="session-rename-cancel"
                    onClick={cancelRename}
                    disabled={disabled}
                  >
                    Cancel
                  </button>
                </div>
              </form>
            ) : (
              <>
                <Link
                  className={
                    s.id === activeId ? 'session-link active' : 'session-link'
                  }
                  to={`/chat/${s.id}`}
                >
                  <span className="session-title">{s.title}</span>
                  <span className="session-meta">{s.mode}</span>
                </Link>
                <div className="session-actions">
                  <button
                    type="button"
                    className="session-rename"
                    onClick={(e) => {
                      e.preventDefault()
                      e.stopPropagation()
                      beginRename(s)
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
                      e.stopPropagation()
                      onDeleteSession(s.id)
                    }}
                    disabled={disabled}
                    aria-label={`Delete chat ${s.title}`}
                    title="Delete chat"
                  >
                    ×
                  </button>
                </div>
              </>
            )}
          </li>
        ))}
      </ul>
    </aside>
  )
}
