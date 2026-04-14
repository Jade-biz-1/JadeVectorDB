import { useState, useEffect } from 'react';
import Layout from '../components/Layout';
import { usersApi, authApi } from '../lib/api';
import {
  Alert, AlertDescription,
  Button,
  Card, CardHeader, CardTitle, CardDescription, CardContent,
  EmptyState,
  FormField,
  LoadingSpinner,
  Modal,
  StatusBadge,
} from '../components/ui';

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

  const inputCls = 'w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300 focus:border-indigo-500 transition';

  return (
    <Layout title="User Management - JadeVectorDB">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-1">User Management</h1>
        <p className="text-gray-500">Create and manage user accounts</p>
      </div>

      {error && (
        <Alert variant="destructive" className="mb-6 bg-red-50 border-red-200 text-red-800">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
      {success && (
        <Alert className="mb-6 bg-green-50 border-green-200 text-green-800">
          <AlertDescription>{success}</AlertDescription>
        </Alert>
      )}

      {/* ── Add / Edit form ── */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-xl">{editingId ? 'Edit User' : 'Add New User'}</CardTitle>
          <CardDescription>
            {editingId ? 'Update user information' : 'Create a new user account'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={editingId ? handleUpdateUser : handleAddUser}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">

              <FormField label="Username" htmlFor="username" required>
                <input
                  type="text"
                  id="username"
                  name="username"
                  className={inputCls}
                  value={form.username}
                  onChange={handleInputChange}
                  required
                  placeholder="john_doe"
                />
              </FormField>

              <FormField label="Email" htmlFor="email" required>
                <input
                  type="email"
                  id="email"
                  name="email"
                  className={inputCls}
                  value={form.email}
                  onChange={handleInputChange}
                  required
                  placeholder="john@example.com"
                />
              </FormField>

              <FormField
                label={`Password${editingId ? ' (leave blank to keep current)' : ''}`}
                htmlFor="password"
                required={!editingId}
              >
                <input
                  type="password"
                  id="password"
                  name="password"
                  className={inputCls}
                  value={form.password}
                  onChange={handleInputChange}
                  required={!editingId}
                  placeholder={editingId ? 'Leave blank to keep current' : 'Enter password'}
                />
              </FormField>

              <FormField label="Roles (comma-separated)" htmlFor="roles">
                <input
                  type="text"
                  id="roles"
                  name="roles"
                  className={inputCls}
                  value={form.roles}
                  onChange={handleInputChange}
                  placeholder="admin, developer, user"
                />
              </FormField>
            </div>

            <div className="flex gap-3 mt-5">
              <Button type="submit" disabled={saving}>
                {editingId ? (saving ? 'Updating…' : 'Update User') : (saving ? 'Adding…' : 'Add User')}
              </Button>
              {editingId && (
                <Button
                  type="button"
                  variant="secondary"
                  onClick={() => {
                    setEditingId(null);
                    setForm({ username: '', password: '', email: '', roles: '' });
                  }}
                >
                  Cancel
                </Button>
              )}
            </div>
          </form>
        </CardContent>
      </Card>

      {/* ── Users table ── */}
      <Card>
        <CardHeader>
          <CardTitle className="text-xl">Users</CardTitle>
          <CardDescription>Manage existing user accounts</CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <LoadingSpinner label="Loading users…" />
          ) : users.length === 0 ? (
            <EmptyState
              icon="👤"
              title="No users found"
              description="Add your first user above"
            />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-sm">
                <thead>
                  <tr className="bg-gray-50">
                    <th className="text-left px-4 py-3 border-b-2 border-gray-200 text-xs font-semibold text-gray-500 uppercase tracking-wide">User ID</th>
                    <th className="text-left px-4 py-3 border-b-2 border-gray-200 text-xs font-semibold text-gray-500 uppercase tracking-wide">Username</th>
                    <th className="text-left px-4 py-3 border-b-2 border-gray-200 text-xs font-semibold text-gray-500 uppercase tracking-wide">Email</th>
                    <th className="text-left px-4 py-3 border-b-2 border-gray-200 text-xs font-semibold text-gray-500 uppercase tracking-wide">Roles</th>
                    <th className="text-left px-4 py-3 border-b-2 border-gray-200 text-xs font-semibold text-gray-500 uppercase tracking-wide">Status</th>
                    <th className="text-left px-4 py-3 border-b-2 border-gray-200 text-xs font-semibold text-gray-500 uppercase tracking-wide">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {users.map(user => (
                    <tr key={user.id} className="border-b border-gray-100 hover:bg-gray-50 transition-colors">
                      <td className="px-4 py-3 text-gray-700">{user.id}</td>
                      <td className="px-4 py-3 font-medium text-gray-900">{user.username}</td>
                      <td className="px-4 py-3 text-gray-600">{user.email}</td>
                      <td className="px-4 py-3 text-gray-600">
                        {Array.isArray(user.roles) ? user.roles.join(', ') : user.roles}
                      </td>
                      <td className="px-4 py-3">
                        <StatusBadge status={user.status || 'active'} />
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex gap-2 flex-wrap">
                          <Button size="sm" onClick={() => handleEditUser(user)} disabled={saving}>
                            Edit
                          </Button>
                          <Button size="sm" variant="outline" onClick={() => handleOpenResetPassword(user)} disabled={saving}>
                            Reset Password
                          </Button>
                          <Button size="sm" variant="destructive" onClick={() => handleDeleteUser(user.id)} disabled={saving}>
                            Delete
                          </Button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* ── Reset Password Modal ── */}
      <Modal
        open={resetPasswordModal.show}
        onClose={handleCloseResetPassword}
        title="Reset Password"
      >
        <p className="text-sm text-gray-500 mb-4">
          Resetting password for: <strong className="text-gray-800">{resetPasswordModal.username}</strong>
        </p>

        <form onSubmit={handleResetPassword}>
          <FormField
            label="New Password"
            htmlFor="newPassword"
            required
            hint="Must be at least 10 characters with uppercase, lowercase, digit, and special character"
          >
            <input
              type="password"
              id="newPassword"
              className={inputCls}
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              required
              placeholder="Enter new password"
              disabled={saving}
            />
          </FormField>

          {error && (
            <Alert variant="destructive" className="mt-3 bg-red-50 border-red-200 text-red-800">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <Alert className="mt-3 bg-amber-50 border-amber-200 text-amber-800">
            <AlertDescription>
              User will be required to change this password on their next login.
            </AlertDescription>
          </Alert>

          <div className="flex justify-end gap-3 mt-5">
            <Button type="button" variant="secondary" onClick={handleCloseResetPassword} disabled={saving}>
              Cancel
            </Button>
            <Button type="submit" disabled={saving}>
              {saving ? 'Resetting…' : 'Reset Password'}
            </Button>
          </div>
        </form>
      </Modal>
    </Layout>
  );
}
