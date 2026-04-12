import { createContext, useContext, useEffect, useMemo, useState } from 'react'

export const THEME_STORAGE_KEY = 'neural-tutor-theme'

const ThemeContext = createContext(null)

function readTheme() {
  if (typeof window === 'undefined') return 'light'
  const fromDom = document.documentElement.dataset.theme
  if (fromDom === 'dark' || fromDom === 'light') return fromDom
  try {
    const s = localStorage.getItem(THEME_STORAGE_KEY)
    if (s === 'dark' || s === 'light') return s
  } catch {
    /* ignore */
  }
  return 'light'
}

export function ThemeProvider({ children }) {
  const [theme, setThemeState] = useState(readTheme)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try {
      localStorage.setItem(THEME_STORAGE_KEY, theme)
    } catch {
      /* ignore */
    }
  }, [theme])

  const value = useMemo(
    () => ({
      theme,
      toggleTheme: () => setThemeState((t) => (t === 'dark' ? 'light' : 'dark')),
      setTheme: setThemeState,
    }),
    [theme]
  )

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export function useTheme() {
  const ctx = useContext(ThemeContext)
  if (!ctx) {
    throw new Error('useTheme must be used within ThemeProvider')
  }
  return ctx
}
