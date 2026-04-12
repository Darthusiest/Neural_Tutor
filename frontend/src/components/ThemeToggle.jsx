import { useTheme } from '../context/ThemeContext'

export function ThemeToggle({ className = '' }) {
  const { theme, toggleTheme } = useTheme()
  const next = theme === 'dark' ? 'light' : 'dark'
  const label =
    theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'

  return (
    <button
      type="button"
      className={`theme-toggle ${className}`.trim()}
      onClick={toggleTheme}
      aria-label={label}
      title={label}
    >
      <span className="theme-toggle-icon" aria-hidden>
        {theme === 'dark' ? '☀️' : '🌙'}
      </span>
      <span className="theme-toggle-text">{next === 'dark' ? 'Dark' : 'Light'}</span>
    </button>
  )
}
