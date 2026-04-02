function QueryResponse({ response }) {
  if (!response || !response.success) {
    return null;
  }

  const formatAnswer = (text) => {
    // Simple formatting: convert numbered lists and bullet points
    return text.split('\n').map((line, i) => {
      const trimmed = line.trim();

      // Numbered items
      if (/^\d+\./.test(trimmed)) {
        return <li key={i} className="numbered-item">{trimmed.substring(trimmed.indexOf('.') + 1).trim()}</li>;
      }

      // Bullet points
      if (trimmed.startsWith('-') || trimmed.startsWith('•')) {
        return <li key={i} className="bullet-item">{trimmed.substring(1).trim()}</li>;
      }

      // Bold text (wrapped in **)
      if (trimmed.includes('**')) {
        const parts = trimmed.split('**');
        return (
          <p key={i}>
            {parts.map((part, j) =>
              j % 2 === 1 ? <strong key={j}>{part}</strong> : part
            )}
          </p>
        );
      }

      // Regular paragraphs
      if (trimmed) {
        return <p key={i}>{trimmed}</p>;
      }

      return null;
    }).filter(Boolean);
  };

  return (
    <div className="query-response">
      <div className="response-header">
        <h3>Answer</h3>
        <div className="response-meta">
          <span className="confidence-badge" data-level={response.confidence > 0.8 ? 'high' : response.confidence > 0.6 ? 'medium' : 'low'}>
            Confidence: {(response.confidence * 100).toFixed(0)}%
          </span>
          <span className="mode-badge" data-mode={response.mode}>
            {response.mode}
          </span>
          <span className="time-badge">
            {response.processing_time_ms}ms
          </span>
        </div>
      </div>

      <div className="response-answer">
        {formatAnswer(response.answer)}
      </div>

      {response.sources && response.sources.length > 0 && (
        <div className="response-sources">
          <h4>Sources</h4>
          <div className="sources-list">
            {response.sources.map((source, i) => (
              <div key={i} className="source-card">
                <div className="source-header">
                  <span className="source-doc">{source.doc_name}</span>
                  <span className="source-relevance">
                    {(source.relevance * 100).toFixed(0)}% relevant
                  </span>
                </div>
                <div className="source-details">
                  <span className="source-page">Page {source.page_numbers}</span>
                  {source.section && (
                    <span className="source-section">{source.section}</span>
                  )}
                </div>
                <p className="source-excerpt">{source.excerpt}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default QueryResponse;
