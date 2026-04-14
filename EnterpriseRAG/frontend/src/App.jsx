import { BrowserRouter as Router, Routes, Route, Link, Navigate, useNavigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import QueryPage from './pages/QueryPage';
import AdminPage from './pages/AdminPage';
import AnalyticsPage from './pages/AnalyticsPage';
import LoginPage from './pages/LoginPage';
import ChangePasswordPage from './pages/ChangePasswordPage';
import UsersPage from './pages/UsersPage';
import './styles/App.css';

/**
 * Redirect to /login if not authenticated.
 * Redirect to /change-password if the flag is set (unless already going there).
 */
function RequireAuth({ children, allowWithPendingPassword = false }) {
  const { user } = useAuth();
  if (!user) return <Navigate to="/login" replace />;
  if (user.mustChangePassword && !allowWithPendingPassword) {
    return <Navigate to="/change-password" replace />;
  }
  return children;
}

/** Require admin role. */
function RequireAdmin({ children }) {
  const { user, isAdmin } = useAuth();
  if (!user) return <Navigate to="/login" replace />;
  if (!isAdmin) return <Navigate to="/" replace />;
  return children;
}

function NavBar() {
  const { user, isAdmin, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login', { replace: true });
  };

  if (!user) return null;

  return (
    <nav className="navbar">
      <div className="nav-container">
        <h1 className="nav-logo">EnterpriseRAG</h1>
        <div className="nav-links">
          <Link to="/" className="nav-link">Query</Link>
          <Link to="/admin/documents" className="nav-link">Documents</Link>
          <Link to="/analytics" className="nav-link">Analytics</Link>
          {isAdmin && <Link to="/admin/users" className="nav-link">Users</Link>}
        </div>
        <div className="nav-user">
          <span className="nav-username">{user.username}</span>
          <Link to="/change-password" className="nav-link nav-link-sm">Change Password</Link>
          <button className="nav-logout" onClick={handleLogout}>Sign Out</button>
        </div>
      </div>
    </nav>
  );
}

function AppShell() {
  return (
    <div className="app">
      <NavBar />

      <main className="main-content">
        <Routes>
          {/* Public */}
          <Route path="/login" element={<LoginPage />} />

          {/* Authenticated — allow even when mustChangePassword is set */}
          <Route path="/change-password" element={
            <RequireAuth allowWithPendingPassword>
              <ChangePasswordPage />
            </RequireAuth>
          } />
          <Route path="/" element={
            <RequireAuth><QueryPage /></RequireAuth>
          } />
          <Route path="/admin/documents" element={
            <RequireAuth><AdminPage /></RequireAuth>
          } />
          <Route path="/analytics" element={
            <RequireAuth><AnalyticsPage /></RequireAuth>
          } />

          {/* Admin only */}
          <Route path="/admin/users" element={
            <RequireAdmin><UsersPage /></RequireAdmin>
          } />

          {/* Fallback */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>

      <footer className="footer">
        <p>EnterpriseRAG v1.0.0 - Document Q&A System</p>
      </footer>
    </div>
  );
}

function App() {
  return (
    <Router>
      <AuthProvider>
        <AppShell />
      </AuthProvider>
    </Router>
  );
}

export default App;
