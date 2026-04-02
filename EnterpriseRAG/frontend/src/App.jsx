import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import QueryPage from './pages/QueryPage';
import AdminPage from './pages/AdminPage';
import AnalyticsPage from './pages/AnalyticsPage';
import './styles/App.css';

function App() {
  return (
    <Router>
      <div className="app">
        <nav className="navbar">
          <div className="nav-container">
            <h1 className="nav-logo">EnterpriseRAG</h1>
            <div className="nav-links">
              <Link to="/" className="nav-link">Query</Link>
              <Link to="/admin/documents" className="nav-link">Documents</Link>
              <Link to="/analytics" className="nav-link">Analytics</Link>
            </div>
          </div>
        </nav>

        <main className="main-content">
          <Routes>
            <Route path="/" element={<QueryPage />} />
            <Route path="/admin/documents" element={<AdminPage />} />
            <Route path="/analytics" element={<AnalyticsPage />} />
          </Routes>
        </main>

        <footer className="footer">
          <p>EnterpriseRAG v1.0.0 - Maintenance Documentation Q&A System</p>
        </footer>
      </div>
    </Router>
  );
}

export default App;
