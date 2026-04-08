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

/**
 * Query Interface API
 */
export const queryAPI = {
  /**
   * Submit a question to the RAG system
   */
  async query(question, deviceType = 'all', topK = 5) {
    const response = await api.post('/api/query', {
      question,
      device_type: deviceType,
      top_k: topK,
    });
    return response.data;
  },

  /**
   * Get system statistics
   */
  async getStats() {
    const response = await api.get('/api/stats');
    return response.data;
  },

  /**
   * Health check
   */
  async healthCheck() {
    const response = await api.get('/api/health');
    return response.data;
  },

  /**
   * Get query analytics
   */
  async getAnalytics(recent = 20) {
    const response = await api.get('/api/analytics', { params: { recent } });
    return response.data;
  },
};

/**
 * Admin Interface API
 */
export const adminAPI = {
  /**
   * Upload a document
   */
  async uploadDocument(file, deviceType) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('device_type', deviceType);

    const response = await api.post('/api/admin/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  /**
   * List all documents
   */
  async listDocuments() {
    const response = await api.get('/api/admin/documents');
    return response.data;
  },

  /**
   * Get document processing status
   */
  async getDocumentStatus(docId) {
    const response = await api.get(`/api/admin/documents/${docId}/status`);
    return response.data;
  },

  /**
   * Delete a document
   */
  async deleteDocument(docId) {
    const response = await api.delete(`/api/admin/documents/${docId}`);
    return response.data;
  },

  /**
   * Reprocess a document
   */
  async reprocessDocument(docId) {
    const response = await api.post(`/api/admin/documents/${docId}/reprocess`);
    return response.data;
  },
};

export default api;
