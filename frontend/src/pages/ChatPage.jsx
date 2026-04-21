import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { apiFetch } from '../api/client'
import { Sidebar } from '../components/Sidebar'
import { ChatPanel } from '../components/ChatPanel'

export function ChatPage({ user }) {
  const { sessionId } = useParams()
  const nav = useNavigate()
  const [sessions, setSessions] = useState([])
  const [messages, setMessages] = useState([])
  const [boostEnabled, setBoostEnabled] = useState(false)
  const [mode, setMode] = useState('auto')
  const [lastModeRouting, setLastModeRouting] = useState(null)
  const [sending, setSending] = useState(false)

  const activeId = sessionId ? parseInt(sessionId, 10) : null

  async function refreshSessions() {
    const data = await apiFetch('/api/sessions')
    setSessions(data.sessions || [])
  }

  async function refreshMessages(sid) {
    if (!sid) {
      setMessages([])
      return
    }
    const data = await apiFetch(`/api/sessions/${sid}/messages`)
    const msgs = (data.messages || []).map((m) => {
      if (m.role === 'user') {
        return { id: m.id, role: 'user', content: m.content || '' }
      }
      return {
        id: m.id,
        role: 'assistant',
        course_answer: stripPrefix(m.course_answer, 'Course Answer:'),
        boosted_explanation: m.boosted_explanation
          ? stripPrefix(m.boosted_explanation, 'Boosted Explanation:')
          : null,
        study: m.study,
      }
    })
    setMessages(msgs)
  }

  useEffect(() => {
    if (!user) return
    refreshSessions().catch(() => {})
  }, [user])

  useEffect(() => {
    if (!user || !activeId) return
    refreshMessages(activeId).catch(() => {})
  }, [user, activeId])

  async function handleNewChat() {
    const data = await apiFetch('/api/sessions', {
      method: 'POST',
      body: JSON.stringify({ title: 'New chat', mode: 'auto' }),
    })
    await refreshSessions()
    nav(`/chat/${data.session.id}`)
  }

  async function ensureSession() {
    if (activeId) return activeId
    const created = await apiFetch('/api/sessions', {
      method: 'POST',
      body: JSON.stringify({
        title: 'Study session',
        mode,
      }),
    })
    const sid = created.session.id
    nav(`/chat/${sid}`)
    await refreshSessions()
    return sid
  }

  async function refreshAfterStudy(sid) {
    await refreshMessages(sid)
    await refreshSessions()
  }

  async function handleRenameSession(sid, newTitle) {
    const title = (newTitle || '').trim() || 'New chat'
    await apiFetch(`/api/sessions/${sid}`, {
      method: 'PATCH',
      body: JSON.stringify({ title }),
    })
    await refreshSessions()
  }

  async function handleDeleteSession(sid) {
    if (!window.confirm('Delete this chat? This cannot be undone.')) return
    try {
      await apiFetch(`/api/sessions/${sid}`, { method: 'DELETE' })
      setSessions((prev) => prev.filter((s) => s.id !== sid))
      if (activeId === sid) {
        setMessages([])
        nav('/chat')
      }
    } catch {
      await refreshSessions().catch(() => {})
    }
  }

  async function handleSend(text) {
    let sid = activeId
    if (!sid) {
      const created = await apiFetch('/api/sessions', {
        method: 'POST',
        body: JSON.stringify({
          title: (text.slice(0, 48) || 'New chat').trim(),
          mode,
        }),
      })
      sid = created.session.id
      nav(`/chat/${sid}`)
      await refreshSessions()
    }
    setSending(true)
    const optimistic = [
      ...messages,
      { id: `tmp-${Date.now()}`, role: 'user', content: text },
    ]
    setMessages(optimistic)
    try {
      const chatPayload = {
        session_id: sid,
        message: text,
        boost_toggle: boostEnabled,
      }
      if (mode === 'auto') {
        // Server detects from message; omit mode keys (optional contract).
      } else {
        chatPayload.mode_override = mode
      }
      const data = await apiFetch('/api/chat', {
        method: 'POST',
        body: JSON.stringify(chatPayload),
      })
      if (data?.mode) {
        setLastModeRouting(data.mode)
      } else if (data?.mode_routing) {
        setLastModeRouting(data.mode_routing)
      }
      await refreshMessages(sid)
      await refreshSessions()
    } catch (e) {
      await refreshMessages(sid)
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="chat-layout">
      <Sidebar
        sessions={sessions}
        activeId={activeId}
        onNewChat={handleNewChat}
        onRenameSession={handleRenameSession}
        onDeleteSession={handleDeleteSession}
        disabled={sending}
      />
      <ChatPanel
        messages={messages}
        boostEnabled={boostEnabled}
        onBoostChange={setBoostEnabled}
        onSend={handleSend}
        sending={sending}
        mode={mode}
        onModeChange={setMode}
        lastModeRouting={lastModeRouting}
        ensureSession={ensureSession}
        onRefreshAfterStudy={refreshAfterStudy}
      />
    </div>
  )
}

function stripPrefix(text, prefix) {
  if (!text) return ''
  const t = text.trimStart()
  if (t.toLowerCase().startsWith(prefix.toLowerCase())) {
    return t.slice(prefix.length).trimStart()
  }
  return text
}
