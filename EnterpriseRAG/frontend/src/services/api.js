/**
 * API service for EnterpriseRAG backend
 */

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Attach stored JWT to every request
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Redirect to /login on 401
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('auth_token');
      localStorage.removeItem('auth_user');
      // Avoid redirect loop on the login page itself
      if (window.location.pathname !== '/login') {
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

/**
 * Auth API
 */
export const authAPI = {
  async login(username, password) {
    const response = await api.post('/api/auth/login', { username, password });
    return response.data;
  },

  async changePassword(currentPassword, newPassword) {
    const response = await api.post('/api/auth/change-password', {
      current_password: currentPassword,
      new_password: newPassword,
    });
    return response.data;
  },
};

/**
 * Users API (admin only)
 */
export const usersAPI = {
  async listUsers(offset = 0, limit = 50) {
    const response = await api.get('/api/users', { params: { offset, limit } });
    return response.data;
  },

  async createUser(username, email, role = 'user') {
    const response = await api.post('/api/users', { username, email, role });
    return response.data;
  },

  async deleteUser(userId) {
    const response = await api.delete(`/api/users/${userId}`);
    return response.data;
  },

  async resetPassword(userId) {
    const response = await api.post(`/api/users/${userId}/reset-password`);
    return response.data;
  },
};

/**
 * Query Interface API
 */
export const queryAPI = {
  async query(question, category = 'all', topK = 5) {
    const response = await api.post('/api/query', {
      question,
      category,
      top_k: topK,
    });
    return response.data;
  },

  async getStats() {
    const response = await api.get('/api/stats');
    return response.data;
  },

  async healthCheck() {
    const response = await api.get('/api/health');
    return response.data;
  },

  async getAnalytics(recent = 20) {
    const response = await api.get('/api/analytics', { params: { recent } });
    return response.data;
  },
};

/**
 * Admin Interface API
 */
export const adminAPI = {
  async uploadDocument(file, category) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('category', category);

    const response = await api.post('/api/admin/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  async listDocuments() {
    const response = await api.get('/api/admin/documents');
    return response.data;
  },

  async getDocumentStatus(docId) {
    const response = await api.get(`/api/admin/documents/${docId}/status`);
    return response.data;
  },

  async deleteDocument(docId) {
    const response = await api.delete(`/api/admin/documents/${docId}`);
    return response.data;
  },

  async reprocessDocument(docId) {
    const response = await api.post(`/api/admin/documents/${docId}/reprocess`);
    return response.data;
  },
};

export default api;
