function SystemStats({ stats }) {
  if (!stats) {
    return null;
  }

  const formatUptime = (seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  return (
    <div className="system-stats">
      <h4>System Statistics</h4>
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-value">{stats.total_documents}</div>
          <div className="stat-label">Documents</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{stats.total_chunks.toLocaleString()}</div>
          <div className="stat-label">Chunks</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{stats.total_queries}</div>
          <div className="stat-label">Queries</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{formatUptime(stats.uptime_seconds)}</div>
          <div className="stat-label">Uptime</div>
        </div>
      </div>
      <div className="stats-status">
        <span className={`status-indicator ${stats.status}`}>{stats.status}</span>
        <span className="db-status">DB: {stats.db_status}</span>
        <span className="llm-status">LLM: {stats.llm_status}</span>
      </div>
    </div>
  );
}

export default SystemStats;
