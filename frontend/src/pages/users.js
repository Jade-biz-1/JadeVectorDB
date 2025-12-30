import { useState, useEffect } from 'react';
import Layout from '../components/Layout';
import { usersApi, authApi } from '../lib/api';

export default function UserManagement() {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [form, setForm] = useState({ username: '', password: '', email: '', roles: '' });
  const [editingId, setEditingId] = useState(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [resetPasswordModal, setResetPasswordModal] = useState({ show: false, userId: null, username: '' });
  const [newPassword, setNewPassword] = useState('');

  const fetchUsers = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await usersApi.listUsers();
      setUsers(response.users || []);
    } catch (error) {
      console.error('Error fetching users:', error);
      setError(`Error fetching users: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchUsers();
  }, []);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setForm({ ...form, [name]: value });
  };

  const handleAddUser = async (e) => {
    e.preventDefault();
    setSaving(true);
    setError('');
    setSuccess('');
    try {
      const rolesArray = form.roles ? form.roles.split(',').map(r => r.trim()) : [];
      await usersApi.createUser(
        form.username,
        form.password,
        form.email,
        rolesArray
      );
      setForm({ username: '', password: '', email: '', roles: '' });
      setSuccess('User created successfully!');
      fetchUsers();
    } catch (error) {
      console.error('Error adding user:', error);
      setError(`Error adding user: ${error.message}`);
    } finally {
      setSaving(false);
    }
  };

  const handleEditUser = (user) => {
    setEditingId(user.user_id);
    setForm({
      username: user.username,
      password: '', // Don't populate password for security
      email: user.email || '',
      roles: Array.isArray(user.roles) ? user.roles.join(', ') : user.roles || ''
    });
  };

  const handleUpdateUser = async (e) => {
    e.preventDefault();
    setSaving(true);
    setError('');
    setSuccess('');
    try {
      const rolesArray = form.roles ? form.roles.split(',').map(r => r.trim()) : [];
      const updateData = {
        username: form.username,
        email: form.email,
        roles: rolesArray
      };
      // Only include password if it's been entered
      if (form.password) {
        updateData.password = form.password;
      }
      await usersApi.updateUser(editingId, updateData);
      setEditingId(null);
      setForm({ username: '', password: '', email: '', roles: '' });
      setSuccess('User updated successfully!');
      fetchUsers();
    } catch (error) {
      console.error('Error updating user:', error);
      setError(`Error updating user: ${error.message}`);
    } finally {
      setSaving(false);
    }
  };

  const handleDeleteUser = async (userId) => {
    if (!window.confirm('Are you sure you want to delete this user? This action cannot be undone.')) {
      return;
    }
    setSaving(true);
    setError('');
    setSuccess('');
    try {
      await usersApi.deleteUser(userId);
      setSuccess('User deleted successfully!');
      fetchUsers();
    } catch (error) {
      console.error('Error deleting user:', error);
      setError(`Error deleting user: ${error.message}`);
    } finally {
      setSaving(false);
    }
  };

  const handleOpenResetPassword = (user) => {
    setResetPasswordModal({
      show: true,
      userId: user.user_id || user.id,
      username: user.username
    });
    setNewPassword('');
    setError('');
    setSuccess('');
  };

  const handleCloseResetPassword = () => {
    setResetPasswordModal({ show: false, userId: null, username: '' });
    setNewPassword('');
  };

  const handleResetPassword = async (e) => {
    e.preventDefault();
    setSaving(true);
    setError('');
    setSuccess('');

    try {
      // Validate password strength
      if (newPassword.length < 10) {
        setError('Password must be at least 10 characters long');
        setSaving(false);
        return;
      }

      if (!/[A-Z]/.test(newPassword) || !/[a-z]/.test(newPassword) ||
          !/[0-9]/.test(newPassword) || !/[^A-Za-z0-9]/.test(newPassword)) {
        setError('Password must contain uppercase, lowercase, digit, and special character');
        setSaving(false);
        return;
      }

      await authApi.adminResetPassword(resetPasswordModal.userId, newPassword);
      setSuccess(`Password reset successfully for ${resetPasswordModal.username}. User will be required to change password on next login.`);
      handleCloseResetPassword();
      fetchUsers();
    } catch (error) {
      console.error('Error resetting password:', error);
      setError(`Error resetting password: ${error.message}`);
    } finally {
      setSaving(false);
    }
  };

  return (
    <Layout title="User Management - JadeVectorDB">
      <style jsx>{`
        .users-container {
          max-width: 1200px;
          margin: 0 auto;
          padding: 20px;
        }

        .page-header {
          margin-bottom: 30px;
        }

        .page-title {
          font-size: 32px;
          font-weight: 700;
          color: #2c3e50;
          margin: 0 0 10px 0;
        }

        .page-description {
          color: #7f8c8d;
          font-size: 16px;
        }

        .alert {
          padding: 15px;
          border-radius: 8px;
          margin-bottom: 20px;
          font-size: 14px;
        }

        .alert-error {
          background: #fee2e2;
          border: 1px solid #fecaca;
          color: #991b1b;
        }

        .alert-success {
          background: #dcfce7;
          border: 1px solid #bbf7d0;
          color: #166534;
        }

        .card {
          background: white;
          border-radius: 8px;
          padding: 30px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.1);
          margin-bottom: 30px;
        }

        .card-title {
          font-size: 20px;
          font-weight: 600;
          color: #2c3e50;
          margin: 0 0 10px 0;
        }

        .card-subtitle {
          font-size: 14px;
          color: #7f8c8d;
          margin-bottom: 25px;
        }

        .form-grid {
          display: grid;
          grid-template-columns: 1fr;
          gap: 15px;
        }

        @media (min-width: 768px) {
          .form-grid {
            grid-template-columns: 1fr 1fr;
          }
        }

        .form-group {
          display: flex;
          flex-direction: column;
        }

        .form-label {
          font-weight: 500;
          color: #2c3e50;
          margin-bottom: 8px;
          font-size: 14px;
        }

        .form-input {
          padding: 10px 12px;
          border: 1px solid #d1d5db;
          border-radius: 6px;
          font-size: 14px;
          transition: all 0.2s;
        }

        .form-input:focus {
          outline: none;
          border-color: #3498db;
          box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
        }

        .btn {
          padding: 10px 20px;
          border-radius: 6px;
          font-weight: 500;
          font-size: 14px;
          cursor: pointer;
          transition: all 0.2s;
          border: none;
          display: inline-flex;
          align-items: center;
          justify-content: center;
        }

        .btn-primary {
          background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
          color: white;
        }

        .btn-primary:hover:not(:disabled) {
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(52, 152, 219, 0.4);
        }

        .btn-primary:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .btn-secondary {
          background: #95a5a6;
          color: white;
        }

        .btn-secondary:hover:not(:disabled) {
          background: #7f8c8d;
        }

        .btn-danger {
          background: #e74c3c;
          color: white;
          padding: 6px 12px;
          font-size: 12px;
        }

        .btn-danger:hover:not(:disabled) {
          background: #c0392b;
        }

        .btn-edit {
          background: #3498db;
          color: white;
          padding: 6px 12px;
          font-size: 12px;
          margin-right: 8px;
        }

        .btn-edit:hover:not(:disabled) {
          background: #2980b9;
        }

        .btn-warning {
          background: #f39c12;
          color: white;
          padding: 6px 12px;
          font-size: 12px;
          margin-right: 8px;
        }

        .btn-warning:hover:not(:disabled) {
          background: #e67e22;
        }

        .button-group {
          display: flex;
          gap: 10px;
          margin-top: 15px;
        }

        .table-container {
          overflow-x: auto;
        }

        table {
          width: 100%;
          border-collapse: collapse;
        }

        thead {
          background: #f8f9fa;
        }

        th {
          text-align: left;
          padding: 12px 15px;
          border-bottom: 2px solid #ecf0f1;
          font-size: 12px;
          color: #7f8c8d;
          text-transform: uppercase;
          font-weight: 600;
        }

        td {
          padding: 15px;
          border-bottom: 1px solid #ecf0f1;
          font-size: 14px;
          color: #2c3e50;
        }

        tbody tr:hover {
          background: #f8f9fa;
        }

        .badge {
          display: inline-block;
          padding: 4px 12px;
          border-radius: 12px;
          font-size: 12px;
          font-weight: 600;
        }

        .badge-success {
          background: #d4edda;
          color: #155724;
        }

        .badge-error {
          background: #f8d7da;
          color: #721c24;
        }

        .empty-state {
          text-align: center;
          padding: 40px;
          color: #7f8c8d;
        }

        .loading-state {
          text-align: center;
          padding: 40px;
          color: #7f8c8d;
        }

        .modal-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.5);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
        }

        .modal {
          background: white;
          border-radius: 8px;
          padding: 30px;
          max-width: 500px;
          width: 90%;
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }

        .modal-header {
          margin-bottom: 20px;
        }

        .modal-title {
          font-size: 24px;
          font-weight: 600;
          color: #2c3e50;
          margin: 0 0 10px 0;
        }

        .modal-subtitle {
          font-size: 14px;
          color: #7f8c8d;
        }

        .modal-body {
          margin-bottom: 20px;
        }

        .modal-footer {
          display: flex;
          gap: 10px;
          justify-content: flex-end;
        }

        .password-hint {
          font-size: 12px;
          color: #7f8c8d;
          margin-top: 8px;
        }
      `}</style>

      <div className="users-container">
        <div className="page-header">
          <h1 className="page-title">User Management</h1>
          <p className="page-description">Create and manage user accounts</p>
        </div>

        {error && (
          <div className="alert alert-error">
            {error}
          </div>
        )}

        {success && (
          <div className="alert alert-success">
            {success}
          </div>
        )}

        <div className="card">
          <h2 className="card-title">{editingId ? 'Edit User' : 'Add New User'}</h2>
          <p className="card-subtitle">
            {editingId ? 'Update user information' : 'Create a new user account'}
          </p>

          <form onSubmit={editingId ? handleUpdateUser : handleAddUser}>
            <div className="form-grid">
              <div className="form-group">
                <label htmlFor="username" className="form-label">Username *</label>
                <input
                  type="text"
                  id="username"
                  name="username"
                  className="form-input"
                  value={form.username}
                  onChange={handleInputChange}
                  required
                  placeholder="john_doe"
                />
              </div>

              <div className="form-group">
                <label htmlFor="email" className="form-label">Email *</label>
                <input
                  type="email"
                  id="email"
                  name="email"
                  className="form-input"
                  value={form.email}
                  onChange={handleInputChange}
                  required
                  placeholder="john@example.com"
                />
              </div>

              <div className="form-group">
                <label htmlFor="password" className="form-label">
                  Password {editingId ? '(leave blank to keep current)' : '*'}
                </label>
                <input
                  type="password"
                  id="password"
                  name="password"
                  className="form-input"
                  value={form.password}
                  onChange={handleInputChange}
                  required={!editingId}
                  placeholder={editingId ? 'Leave blank to keep current' : 'Enter password'}
                />
              </div>

              <div className="form-group">
                <label htmlFor="roles" className="form-label">Roles (comma-separated)</label>
                <input
                  type="text"
                  id="roles"
                  name="roles"
                  className="form-input"
                  value={form.roles}
                  onChange={handleInputChange}
                  placeholder="admin, developer, user"
                />
              </div>
            </div>

            <div className="button-group">
              <button type="submit" disabled={saving} className="btn btn-primary">
                {editingId ? (saving ? 'Updating...' : 'Update User') : (saving ? 'Adding...' : 'Add User')}
              </button>
              {editingId && (
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => {
                    setEditingId(null);
                    setForm({ username: '', password: '', email: '', roles: '' });
                  }}
                >
                  Cancel
                </button>
              )}
            </div>
          </form>
        </div>

        <div className="card">
          <h2 className="card-title">Users</h2>
          <p className="card-subtitle">Manage existing user accounts</p>

          <div className="table-container">
            {loading ? (
              <div className="loading-state">Loading users...</div>
            ) : users.length === 0 ? (
              <div className="empty-state">No users found. Add your first user above.</div>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>User ID</th>
                    <th>Username</th>
                    <th>Email</th>
                    <th>Roles</th>
                    <th>Status</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {users.map(user => (
                    <tr key={user.id}>
                      <td>{user.id}</td>
                      <td>{user.username}</td>
                      <td>{user.email}</td>
                      <td>{Array.isArray(user.roles) ? user.roles.join(', ') : user.roles}</td>
                      <td>
                        <span className={`badge ${user.status === 'active' ? 'badge-success' : 'badge-error'}`}>
                          {user.status || 'active'}
                        </span>
                      </td>
                      <td>
                        <button
                          className="btn btn-edit"
                          onClick={() => handleEditUser(user)}
                          disabled={saving}
                        >
                          Edit
                        </button>
                        <button
                          className="btn btn-warning"
                          onClick={() => handleOpenResetPassword(user)}
                          disabled={saving}
                        >
                          Reset Password
                        </button>
                        <button
                          className="btn btn-danger"
                          onClick={() => handleDeleteUser(user.id)}
                          disabled={saving}
                        >
                          Delete
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>

        {/* Password Reset Modal */}
        {resetPasswordModal.show && (
          <div className="modal-overlay" onClick={handleCloseResetPassword}>
            <div className="modal" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header">
                <h2 className="modal-title">Reset Password</h2>
                <p className="modal-subtitle">
                  Resetting password for: <strong>{resetPasswordModal.username}</strong>
                </p>
              </div>

              <form onSubmit={handleResetPassword}>
                <div className="modal-body">
                  <div className="form-group">
                    <label htmlFor="newPassword" className="form-label">New Password *</label>
                    <input
                      type="password"
                      id="newPassword"
                      className="form-input"
                      value={newPassword}
                      onChange={(e) => setNewPassword(e.target.value)}
                      required
                      placeholder="Enter new password"
                      disabled={saving}
                    />
                    <p className="password-hint">
                      Must be at least 10 characters with uppercase, lowercase, digit, and special character
                    </p>
                  </div>

                  {error && (
                    <div className="alert alert-error" style={{ marginTop: '15px' }}>
                      {error}
                    </div>
                  )}

                  <div className="alert" style={{
                    background: '#fff3cd',
                    border: '1px solid #ffeaa7',
                    color: '#856404',
                    marginTop: '15px'
                  }}>
                    User will be required to change this password on their next login.
                  </div>
                </div>

                <div className="modal-footer">
                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={handleCloseResetPassword}
                    disabled={saving}
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="btn btn-primary"
                    disabled={saving}
                  >
                    {saving ? 'Resetting...' : 'Reset Password'}
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
}
