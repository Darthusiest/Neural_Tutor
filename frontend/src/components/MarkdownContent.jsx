import ReactMarkdown from 'react-markdown'
import rehypeSanitize from 'rehype-sanitize'
import remarkBreaks from 'remark-breaks'

/**
 * Renders sanitized Markdown for assistant messages (headings, lists, emphasis).
 * XSS-safe: raw HTML is stripped by rehype-sanitize.
 */
export function MarkdownContent({ children }) {
  const text = typeof children === 'string' ? children : ''
  if (!text.trim()) return null
  return (
    <div className="markdown-body">
      <ReactMarkdown remarkPlugins={[remarkBreaks]} rehypePlugins={[rehypeSanitize]}>
        {text}
      </ReactMarkdown>
    </div>
  )
}
