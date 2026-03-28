// RAG UI Alternative #3: FastAPI + React (Frontend)
// Modern React SPA with professional UI
//
// Setup:
//   npm create vite@latest frontend -- --template react
//   cd frontend && npm install
//   npm run dev
//
// API_URL should point to your FastAPI backend

import { useState, useEffect } from 'react'
import './App.css'

const API_URL = 'http://localhost:8000'

function App() {
  const [question, setQuestion] = useState('')
  const [deviceType, setDeviceType] = useState('all')
  const [topK, setTopK] = useState(5)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [stats, setStats] = useState(null)
  const [error, setError] = useState(null)

  // Load stats on mount
  useEffect(() => {
    loadStats()
  }, [])

  // Reload stats after each query
  useEffect(() => {
    if (result) {
      loadStats()
    }
  }, [result])

  const loadStats = async () => {
    try {
      const response = await fetch(`${API_URL}/api/stats`)
      const data = await response.json()
      setStats(data)
    } catch (err) {
      console.error('Failed to load stats:', err)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()

    if (!question.trim()) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch(`${API_URL}/api/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          question: question,
          device_type: deviceType,
          top_k: topK
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const setExample = (exampleQuestion, exampleDevice) => {
    setQuestion(exampleQuestion)
    setDeviceType(exampleDevice)
  }

  return (
    <div className="app">
      <header className="header">
        <div className="container">
          <h1>🔧 Maintenance Documentation Q&A</h1>
          <p className="subtitle">
            <span className="status-dot"></span>
            Ask questions about equipment maintenance, troubleshooting, and technical procedures
          </p>
        </div>
      </header>

      <main className="container">
        <div className="main-grid">
          {/* Left Column: Input Form */}
          <div className="card">
            <h2 className="card-title">Ask a Question</h2>

            <form onSubmit={handleSubmit}>
              <div className="form-group">
                <label htmlFor="question">Your Question</label>
                <textarea
                  id="question"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Example: How do I perform routine maintenance on the hydraulic pump?"
                  rows="5"
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="deviceType">Device Type</label>
                <select
                  id="deviceType"
                  value={deviceType}
                  onChange={(e) => setDeviceType(e.target.value)}
                >
                  <option value="all">All Devices</option>
                  <option value="hydraulic_pump">Hydraulic Pump</option>
                  <option value="air_compressor">Air Compressor</option>
                  <option value="generator">Generator</option>
                  <option value="cnc_machine">CNC Machine</option>
                  <option value="conveyor_belt">Conveyor Belt</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="topK">Number of Sources: {topK}</label>
                <input
                  type="range"
                  id="topK"
                  min="1"
                  max="15"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                />
              </div>

              <button type="submit" disabled={loading || !question.trim()}>
                {loading ? '🔄 Searching...' : '🔍 Search Documentation'}
              </button>
            </form>

            <div className="examples">
              <p className="examples-label">Try these examples:</p>
              <button
                type="button"
                className="example-btn"
                onClick={() => setExample('How do I replace the air filter?', 'air_compressor')}
              >
                Air filter replacement
              </button>
              <button
                type="button"
                className="example-btn"
                onClick={() => setExample('Safety precautions for hydraulic systems', 'hydraulic_pump')}
              >
                Safety precautions
              </button>
              <button
                type="button"
                className="example-btn"
                onClick={() => setExample('Troubleshooting error code E47', 'cnc_machine')}
              >
                Error code E47
              </button>
            </div>

            {stats && (
              <div className="stats">
                <div className="stat">
                  <div className="stat-value">{stats.total_queries}</div>
                  <div className="stat-label">Queries</div>
                </div>
                <div className="stat">
                  <div className="stat-value">{stats.total_documents}</div>
                  <div className="stat-label">Documents</div>
                </div>
                <div className="stat">
                  <div className="stat-value">{(stats.total_chunks / 1000).toFixed(1)}K</div>
                  <div className="stat-label">Chunks</div>
                </div>
              </div>
            )}
          </div>

          {/* Right Column: Results */}
          <div className="card results-card">
            <h2 className="card-title">Answer</h2>

            {!loading && !result && !error && (
              <div className="empty-state">
                <svg className="empty-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"></path>
                </svg>
                <p>Ask a question to get started</p>
                <p className="empty-subtitle">Your answer will appear here with relevant source citations</p>
              </div>
            )}

            {loading && (
              <div className="loading-state">
                <div className="spinner"></div>
                <p>Searching documentation...</p>
                <p className="loading-subtitle">This usually takes 2-4 seconds</p>
              </div>
            )}

            {error && (
              <div className="error-state">
                <svg className="error-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                </svg>
                <p>Error: {error}</p>
                <p className="error-subtitle">Please try again or check your connection</p>
              </div>
            )}

            {result && (
              <div className="result-content">
                <div className="answer">
                  <div className="answer-meta">
                    <span className="confidence">Confidence: {(result.confidence * 100).toFixed(0)}%</span>
                    <span className="processing-time">{result.processing_time_ms}ms</span>
                  </div>
                  <div className="answer-text">{result.answer}</div>
                </div>

                <div className="sources">
                  <h3 className="sources-title">📚 Source Documents</h3>
                  {result.sources.map((source, index) => (
                    <div key={index} className="source-card">
                      <div className="source-header">
                        <div className="source-title">{index + 1}. {source.doc_name}</div>
                        <div className="relevance">{(source.relevance * 100).toFixed(0)}%</div>
                      </div>
                      <div className="source-meta">
                        📄 Pages: {source.page_numbers} | 📑 {source.section}
                      </div>
                      <div className="source-excerpt">"{source.excerpt}"</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="footer">
        <p>
          <strong>Status:</strong> <span className="status-dot"></span>
          Connected to JadeVectorDB | 🤖 Ollama LLM | 🔌 Offline Mode
        </p>
        <p className="footer-credits">Powered by JadeVectorDB + FastAPI + React</p>
      </footer>
    </div>
  )
}

export default App
