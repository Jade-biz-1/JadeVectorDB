import { useState, useEffect, useCallback } from 'react';
import { usersAPI } from '../services/api';
import '../styles/UsersPage.css';

function CopyablePassword({ password }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(password);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // fallback: select the text
    }
  };

  return (
    <span className="copyable-password">
      <code>{password}</code>
      <button className="copy-btn" onClick={handleCopy} title="Copy password">
        {copied ? 'Copied!' : 'Copy'}
      </button>
    </span>
  );
}

function CreateUserModal({ onClose, onCreated }) {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [role, setRole] = useState('user');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [created, setCreated] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const result = await usersAPI.createUser(username, email, role);
      setCreated(result);
      onCreated();
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to create user');
    } finally {
      setLoading(false);
    }
  };

  if (created) {
    return (
      <div className="modal-overlay" onClick={onClose}>
        <div className="modal" onClick={(e) => e.stopPropagation()}>
          <h3>User Created</h3>
          <p>
            <strong>{created.username}</strong> has been created. Share this
            one-time password — it will not be shown again.
          </p>
          <div className="password-reveal">
            <CopyablePassword password={created.generated_password} />
          </div>
          <p className="modal-note">
            The user will be asked to change this password on first login.
          </p>
          <button className="modal-close-btn" onClick={onClose}>
            Done
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3>Create User</h3>
        {error && <div className="auth-error">{error}</div>}
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="new-username">Username</label>
            <input
              id="new-username"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoFocus
              disabled={loading}
              required
              pattern="[a-zA-Z0-9_\-]+"
              title="Letters, numbers, underscores, hyphens only"
            />
          </div>
          <div className="form-group">
            <label htmlFor="new-email">Email</label>
            <input
              id="new-email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              disabled={loading}
              required
            />
          </div>
          <div className="form-group">
            <label htmlFor="new-role">Role</label>
            <select
              id="new-role"
              value={role}
              onChange={(e) => setRole(e.target.value)}
              disabled={loading}
            >
              <option value="user">User</option>
              <option value="admin">Admin</option>
            </select>
          </div>
          <div className="modal-actions">
            <button type="button" className="btn-secondary" onClick={onClose} disabled={loading}>
              Cancel
            </button>
            <button type="submit" className="btn-primary" disabled={loading}>
              {loading ? 'Creating…' : 'Create'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

function ResetPasswordModal({ user, onClose }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleReset = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await usersAPI.resetPassword(user.id);
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to reset password');
    } finally {
      setLoading(false);
    }
  };

  if (result) {
    return (
      <div className="modal-overlay" onClick={onClose}>
        <div className="modal" onClick={(e) => e.stopPropagation()}>
          <h3>Password Reset</h3>
          <p>
            New temporary password for <strong>{user.username}</strong>. Share
            it securely — it will not be shown again.
          </p>
          <div className="password-reveal">
            <CopyablePassword password={result.new_password} />
          </div>
          <p className="modal-note">
            The user must change this password on next login.
          </p>
          <button className="modal-close-btn" onClick={onClose}>
            Done
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3>Reset Password</h3>
        {error && <div className="auth-error">{error}</div>}
        <p>
          Reset password for <strong>{user.username}</strong>? A new temporary
          password will be generated.
        </p>
        <div className="modal-actions">
          <button className="btn-secondary" onClick={onClose} disabled={loading}>
            Cancel
          </button>
          <button className="btn-danger" onClick={handleReset} disabled={loading}>
            {loading ? 'Resetting…' : 'Reset Password'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default function UsersPage() {
  const [users, setUsers] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showCreate, setShowCreate] = useState(false);
  const [resetTarget, setResetTarget] = useState(null);

  const loadUsers = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await usersAPI.listUsers();
      setUsers(data.users);
      setTotal(data.total);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load users');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadUsers(); }, [loadUsers]);

  const handleDeactivate = async (userId) => {
    if (!confirm('Deactivate this user? They will no longer be able to log in.')) return;
    try {
      await usersAPI.deleteUser(userId);
      loadUsers();
    } catch (err) {
      alert(err.response?.data?.detail || 'Failed to deactivate user');
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return '—';
    return new Date(dateString).toLocaleString();
  };

  return (
    <div className="users-page">
      <div className="users-container">
        <header className="page-header">
          <h2>User Management</h2>
          <p>Manage accounts and access for your organization</p>
        </header>

        <div className="users-toolbar">
          <span className="users-count">{total} user{total !== 1 ? 's' : ''}</span>
          <button className="btn-primary" onClick={() => setShowCreate(true)}>
            Create User
          </button>
        </div>

        {error && <div className="error-message"><strong>Error:</strong> {error}</div>}

        {loading ? (
          <div className="loading-message">
            <div className="spinner"></div>
            <p>Loading users…</p>
          </div>
        ) : (
          <div className="users-table-wrap">
            <table className="users-table">
              <thead>
                <tr>
                  <th>Username</th>
                  <th>Email</th>
                  <th>Role</th>
                  <th>Status</th>
                  <th>Last Login</th>
                  <th>Created</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {users.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="empty-row">No users found.</td>
                  </tr>
                ) : (
                  users.map((u) => (
                    <tr key={u.id} className={!u.is_active ? 'row-inactive' : ''}>
                      <td className="col-username">{u.username}</td>
                      <td>{u.email}</td>
                      <td>
                        <span className={`role-badge role-${u.role}`}>{u.role}</span>
                      </td>
                      <td>
                        <span className={`status-badge ${u.is_active ? 'active' : 'inactive'}`}>
                          {u.is_active ? 'Active' : 'Inactive'}
                        </span>
                      </td>
                      <td className="col-date">{formatDate(u.last_login)}</td>
                      <td className="col-date">{formatDate(u.created_at)}</td>
                      <td className="col-actions">
                        {u.is_active && (
                          <>
                            <button
                              className="btn-sm btn-secondary"
                              onClick={() => setResetTarget(u)}
                            >
                              Reset Password
                            </button>
                            <button
                              className="btn-sm btn-danger"
                              onClick={() => handleDeactivate(u.id)}
                            >
                              Deactivate
                            </button>
                          </>
                        )}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {showCreate && (
        <CreateUserModal
          onClose={() => setShowCreate(false)}
          onCreated={loadUsers}
        />
      )}

      {resetTarget && (
        <ResetPasswordModal
          user={resetTarget}
          onClose={() => setResetTarget(null)}
        />
      )}
    </div>
  );
}
