import { useState } from 'react';

function QueryForm({ onSubmit, loading }) {
  const [question, setQuestion] = useState('');
  const [category, setCategory] = useState('all');
  const [topK, setTopK] = useState(5);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (question.trim()) {
      onSubmit(question, category, topK);
    }
  };

  const exampleQuestions = [
    'How do I submit an expense report?',
    'What is the process for requesting time off?',
    'How do I request access to a new system?',
    'What should a new employee do on their first day?',
  ];

  return (
    <div className="query-form-container">
      <form onSubmit={handleSubmit} className="query-form">
        <div className="form-group">
          <label htmlFor="question">Your Question</label>
          <textarea
            id="question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question about your organization's documents..."
            rows="4"
            disabled={loading}
            required
          />
        </div>

        <div className="form-row">
          <div className="form-group">
            <label htmlFor="category">Category</label>
            <select
              id="category"
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              disabled={loading}
            >
              <option value="all">All Categories</option>
              <option value="hr">HR</option>
              <option value="finance">Finance</option>
              <option value="it">IT</option>
              <option value="general">General</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="top-k">Number of Sources</label>
            <select
              id="top-k"
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value))}
              disabled={loading}
            >
              <option value="3">3</option>
              <option value="5">5</option>
              <option value="10">10</option>
            </select>
          </div>
        </div>

        <button
          type="submit"
          className="submit-button"
          disabled={loading || !question.trim()}
        >
          {loading ? 'Processing...' : 'Ask Question'}
        </button>
      </form>

      <div className="example-questions">
        <h4>Example Questions:</h4>
        <div className="example-list">
          {exampleQuestions.map((q, i) => (
            <button
              key={i}
              className="example-button"
              onClick={() => setQuestion(q)}
              disabled={loading}
            >
              {q}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

export default QueryForm;
