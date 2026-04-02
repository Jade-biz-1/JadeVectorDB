import { useState } from 'react';

function QueryForm({ onSubmit, loading }) {
  const [question, setQuestion] = useState('');
  const [deviceType, setDeviceType] = useState('all');
  const [topK, setTopK] = useState(5);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (question.trim()) {
      onSubmit(question, deviceType, topK);
    }
  };

  const exampleQuestions = [
    'How do I replace the hydraulic fluid?',
    'What is the maintenance schedule for the air compressor?',
    'How to troubleshoot overheating issues?',
    'What safety precautions should I take?',
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
            placeholder="Ask a question about maintenance procedures..."
            rows="4"
            disabled={loading}
            required
          />
        </div>

        <div className="form-row">
          <div className="form-group">
            <label htmlFor="device-type">Device Type</label>
            <select
              id="device-type"
              value={deviceType}
              onChange={(e) => setDeviceType(e.target.value)}
              disabled={loading}
            >
              <option value="all">All Devices</option>
              <option value="hydraulic_pump">Hydraulic Pump</option>
              <option value="air_compressor">Air Compressor</option>
              <option value="conveyor">Conveyor System</option>
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
