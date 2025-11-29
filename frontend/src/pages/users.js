import Head from 'next/head';
import { useState, useEffect } from 'react';
import { usersApi } from '../lib/api';

export default function UserManagement() {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [form, setForm] = useState({ username: '', password: '', email: '', roles: '' });
  const [editingId, setEditingId] = useState(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

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

  return (
    <div className="min-h-screen bg-gray-50">
      <Head>
        <title>User Management - JadeVectorDB</title>
        <meta name="description" content="User management dashboard" />
      </Head>
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">User Management</h1>
        </div>
      </header>
      <main>
        <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          <div className="bg-white p-6 rounded-lg shadow mb-8">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Add / Edit User</h2>
            <form onSubmit={editingId ? handleUpdateUser : handleAddUser} className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <input
                type="text"
                name="username"
                placeholder="Username"
                value={form.username}
                onChange={handleInputChange}
                required
                className="border rounded px-3 py-2"
              />
              <input
                type="email"
                name="email"
                placeholder="Email"
                value={form.email}
                onChange={handleInputChange}
                required
                className="border rounded px-3 py-2"
              />
              <input
                type="text"
                name="roles"
                placeholder="Roles (comma separated)"
                value={form.roles}
                onChange={handleInputChange}
                required
                className="border rounded px-3 py-2"
              />
              <select
                name="status"
                value={form.status}
                onChange={handleInputChange}
                className="border rounded px-3 py-2"
              >
                <option value="active">Active</option>
                <option value="inactive">Inactive</option>
              </select>
              <button
                type="submit"
                disabled={saving}
                className="bg-indigo-600 text-white px-4 py-2 rounded mt-2"
              >
                {editingId ? (saving ? 'Updating...' : 'Update User') : (saving ? 'Adding...' : 'Add User')}
              </button>
              {editingId && (
                <button
                  type="button"
                  className="bg-gray-400 text-white px-4 py-2 rounded mt-2"
                  onClick={() => { setEditingId(null); setForm({ username: '', email: '', roles: '', status: 'active' }); }}
                >
                  Cancel
                </button>
              )}
            </form>
          </div>
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Users</h2>
            <table className="min-w-full divide-y divide-gray-200">
              <thead>
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">User ID</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Username</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Email</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Roles</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actions</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {loading ? (
                  <tr><td colSpan={6} className="text-center py-4">Loading...</td></tr>
                ) : users.length === 0 ? (
                  <tr><td colSpan={6} className="text-center py-4">No users found.</td></tr>
                ) : (
                  users.map(user => (
                    <tr key={user.id}>
                      <td className="px-6 py-4 whitespace-nowrap">{user.id}</td>
                      <td className="px-6 py-4 whitespace-nowrap">{user.username}</td>
                      <td className="px-6 py-4 whitespace-nowrap">{user.email}</td>
                      <td className="px-6 py-4 whitespace-nowrap">{Array.isArray(user.roles) ? user.roles.join(', ') : user.roles}</td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${user.status === 'active' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>{user.status}</span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <button className="bg-blue-500 text-white px-2 py-1 rounded mr-2" onClick={() => handleEditUser(user)} disabled={saving}>Edit</button>
                        <button className="bg-red-600 text-white px-2 py-1 rounded" onClick={() => handleDeleteUser(user.id)} disabled={saving}>Delete</button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  );
}
