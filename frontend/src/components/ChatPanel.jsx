import { useEffect, useRef, useState } from 'react'
import { apiFetch } from '../api/client'
import { MarkdownContent } from './MarkdownContent'

function formatModeLabel(m) {
  if (!m) return ''
  const s = String(m)
  return s.charAt(0).toUpperCase() + s.slice(1)
}

export function ChatPanel({
  messages,
  boostEnabled,
  onBoostChange,
  onSend,
  sending,
  overrideEnabled,
  onOverrideEnabledChange,
  overrideMode,
  onOverrideModeChange,
  lastModeRouting,
  ensureSession,
  onRefreshAfterStudy,
}) {
  const messagesEndRef = useRef(null)
  const [quiz, setQuiz] = useState(null)
  const [quizTopic, setQuizTopic] = useState('')
  const [shortText, setShortText] = useState('')
  const [compareA, setCompareA] = useState('')
  const [compareB, setCompareB] = useState('')
  const [compareExpand, setCompareExpand] = useState(false)
  const [summaryKind, setSummaryKind] = useState('lecture')
  const [lectureNum, setLectureNum] = useState('4')
  const [summaryTopic, setSummaryTopic] = useState('')
  const [studyBusy, setStudyBusy] = useState(false)

  const detected =
    lastModeRouting?.detected ?? lastModeRouting?.detected_mode ?? null
  const effective =
    lastModeRouting?.effective ?? lastModeRouting?.effective_mode ?? null
  const ambiguous = lastModeRouting?.ambiguous ?? lastModeRouting?.mode_ambiguous

  useEffect(() => {
    const id = requestAnimationFrame(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
    })
    return () => cancelAnimationFrame(id)
  }, [messages])

  function handleSubmit(e) {
    e.preventDefault()
    const fd = new FormData(e.target)
    const text = (fd.get('message') || '').trim()
    if (!text) return
    onSend(text)
    e.target.reset()
  }

  async function runStudy(body, path) {
    setStudyBusy(true)
    try {
      const sid = await ensureSession()
      const data = await apiFetch(`/api/study/${path}`, {
        method: 'POST',
        body: JSON.stringify({ ...body, session_id: sid }),
      })
      setQuiz(null)
      setShortText('')
      await onRefreshAfterStudy(sid)
      return data
    } finally {
      setStudyBusy(false)
    }
  }

  async function loadQuiz(qt) {
    setStudyBusy(true)
    try {
      const sid = await ensureSession()
      const data = await apiFetch('/api/study/quiz/next', {
        method: 'POST',
        body: JSON.stringify({
          question_type: qt,
          topic: quizTopic.trim() || undefined,
          session_id: sid,
        }),
      })
      setQuiz(data)
    } finally {
      setStudyBusy(false)
    }
  }

  async function submitQuizMc(e) {
    e.preventDefault()
    if (!quiz || quiz.question_type !== 'mc') return
    const fd = new FormData(e.target)
    const sel = fd.get('option')
    if (sel === null || sel === '') return
    await runStudy(
      {
        chunk_id: quiz.chunk_id,
        question_type: 'mc',
        quiz_token: quiz.quiz_token,
        selected_index: parseInt(sel, 10),
        options: quiz.options,
      },
      'quiz/answer'
    )
  }

  async function submitQuizShort(e) {
    e.preventDefault()
    if (!quiz || quiz.question_type !== 'short') return
    const t = shortText.trim()
    if (!t) return
    await runStudy(
      {
        chunk_id: quiz.chunk_id,
        question_type: 'short',
        quiz_token: quiz.quiz_token,
        user_answer: t,
      },
      'quiz/answer'
    )
  }

  async function submitCompare(e) {
    e.preventDefault()
    const a = compareA.trim()
    const b = compareB.trim()
    if (!a || !b) return
    await runStudy(
      {
        concept_a: a,
        concept_b: b,
        expand: compareExpand,
      },
      'compare'
    )
  }

  async function submitSummary(e) {
    e.preventDefault()
    if (summaryKind === 'lecture') {
      const n = parseInt(lectureNum, 10)
      if (Number.isNaN(n)) return
      await runStudy({ kind: 'lecture', lecture_number: n }, 'summary')
    } else {
      const t = summaryTopic.trim()
      if (!t) return
      await runStudy({ kind: 'topic', topic: t }, 'summary')
    }
  }

  const showBoost = !overrideEnabled || overrideMode === 'chat'

  return (
    <section className="chat-panel">
      <div className="chat-toolbar">
        <label className="mode-override-toggle">
          <input
            type="checkbox"
            checked={overrideEnabled}
            onChange={(e) => onOverrideEnabledChange(e.target.checked)}
            disabled={sending || studyBusy}
          />
          Override detected mode
        </label>
        {overrideEnabled ? (
          <label>
            Mode{' '}
            <select
              value={overrideMode}
              onChange={(e) => onOverrideModeChange(e.target.value)}
              disabled={sending || studyBusy}
            >
              <option value="chat">Chat</option>
              <option value="quiz">Quiz</option>
              <option value="compare">Compare</option>
              <option value="summary">Summary</option>
            </select>
          </label>
        ) : null}
        {!overrideEnabled && effective ? (
          <span
            className="muted mode-detected-hint"
            title="Routing from your last assistant reply"
          >
            Detected mode: {formatModeLabel(detected || effective)}
            {detected && effective && detected !== effective
              ? ` · Effective: ${formatModeLabel(effective)}`
              : null}
            {ambiguous ? ' (ambiguous)' : ''}
          </span>
        ) : null}
        {showBoost ? (
          <label className="boost-toggle">
            <input
              type="checkbox"
              checked={boostEnabled}
              onChange={(e) => onBoostChange(e.target.checked)}
            />
            Boosted explanation
          </label>
        ) : null}
      </div>

      <details className="study-tools">
        <summary>Study tools (dedicated quiz, compare, summary)</summary>
        <div className="study-panel">
          <p className="muted">
            One question at a time. After you answer, you&apos;ll see the course-grounded
            explanation. Optional filter:
          </p>
          <input
            type="text"
            placeholder="Topic keyword (optional)"
            value={quizTopic}
            onChange={(e) => setQuizTopic(e.target.value)}
            disabled={studyBusy}
          />
          <div className="study-actions">
            <button type="button" disabled={studyBusy} onClick={() => loadQuiz('mc')}>
              Next question (multiple choice)
            </button>
            <button type="button" disabled={studyBusy} onClick={() => loadQuiz('short')}>
              Next question (short answer)
            </button>
          </div>
          {quiz ? (
            <div className="quiz-card">
              <p className="quiz-question">{quiz.question}</p>
              {quiz.question_type === 'mc' && quiz.options ? (
                <form onSubmit={submitQuizMc}>
                  {quiz.options.map((opt, i) => (
                    <label key={i} className="quiz-option">
                      <input type="radio" name="option" value={i} required />
                      {opt}
                    </label>
                  ))}
                  <button type="submit" disabled={studyBusy}>
                    Submit answer
                  </button>
                </form>
              ) : null}
              {quiz.question_type === 'short' ? (
                <form onSubmit={submitQuizShort}>
                  <textarea
                    rows={3}
                    value={shortText}
                    onChange={(e) => setShortText(e.target.value)}
                    placeholder="Your answer…"
                  />
                  <button type="submit" disabled={studyBusy}>
                    Submit answer
                  </button>
                </form>
              ) : null}
            </div>
          ) : null}
        </div>

        <form className="study-panel compare-form" onSubmit={submitCompare}>
          <p className="muted">Compare two course concepts. Retrieval uses lecture chunks first.</p>
          <input
            placeholder="Concept A (e.g. softmax)"
            value={compareA}
            onChange={(e) => setCompareA(e.target.value)}
            disabled={studyBusy}
          />
          <input
            placeholder="Concept B (e.g. attention)"
            value={compareB}
            onChange={(e) => setCompareB(e.target.value)}
            disabled={studyBusy}
          />
          <label className="boost-toggle">
            <input
              type="checkbox"
              checked={compareExpand}
              onChange={(e) => setCompareExpand(e.target.checked)}
            />
            Expand with GPT (if API key configured)
          </label>
          <button type="submit" disabled={studyBusy}>
            Compare
          </button>
        </form>

        <form className="study-panel" onSubmit={submitSummary}>
          <p className="muted">Summaries are built only from retrieved lecture sections.</p>
          <label>
            <input
              type="radio"
              name="skind"
              checked={summaryKind === 'lecture'}
              onChange={() => setSummaryKind('lecture')}
            />
            By lecture number
          </label>
          <label>
            <input
              type="radio"
              name="skind"
              checked={summaryKind === 'topic'}
              onChange={() => setSummaryKind('topic')}
            />
            By topic / keyword
          </label>
          {summaryKind === 'lecture' ? (
            <input
              type="number"
              min={1}
              max={30}
              value={lectureNum}
              onChange={(e) => setLectureNum(e.target.value)}
            />
          ) : (
            <input
              placeholder="Topic (e.g. MFCC, backpropagation)"
              value={summaryTopic}
              onChange={(e) => setSummaryTopic(e.target.value)}
            />
          )}
          <button type="submit" disabled={studyBusy}>
            Get summary
          </button>
        </form>
      </details>

      <div className="messages">
        {messages.length === 0 ? (
          <p className="muted">
            Ask a question about LING 487 course material. By default the server picks the best mode
            from your message; turn on &quot;Override detected mode&quot; to force Chat, Quiz,
            Compare, or Summary for the main composer.
          </p>
        ) : (
          messages.map((m) => <MessageBlock key={m.id} m={m} />)
        )}
        <div ref={messagesEndRef} className="messages-end" aria-hidden="true" />
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
        <div className="msg-body msg-body--md">
          <MarkdownContent>{m.course_answer || ''}</MarkdownContent>
        </div>
      </div>
      {m.boosted_explanation ? (
        <div className="msg-block boost">
          <div className="label">Boosted Explanation</div>
          <div className="msg-body msg-body--md">
            <MarkdownContent>{m.boosted_explanation}</MarkdownContent>
          </div>
        </div>
      ) : null}
    </div>
  )
}
