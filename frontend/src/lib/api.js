// frontend/src/lib/api.js

// Base API configuration - use proxy to avoid CORS
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || '/api';
const DEFAULT_HEADERS = {
  'Content-Type': 'application/json',
};

// Utility function to handle API responses
const handleResponse = async (response) => {
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.message || `API error: ${response.status}`);
  }
  return response.json();
};

// Utility function to add auth headers
const getAuthHeaders = () => {
  const token = localStorage.getItem('jadevectordb_auth_token');
  if (!token) {
    console.warn('No auth token found in localStorage');
    return DEFAULT_HEADERS;
  }

  return {
    ...DEFAULT_HEADERS,
    'Authorization': `Bearer ${token}`,
  };
};

// API Service for Database Management
export const databaseApi = {
  // List all databases
  listDatabases: async (limit = 20, offset = 0) => {
    const response = await fetch(`${API_BASE_URL}/databases?limit=${limit}&offset=${offset}`, {
      method: 'GET',
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },

  // Create a new database
  createDatabase: async (databaseData) => {
    const response = await fetch(`${API_BASE_URL}/databases`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify(databaseData),
    });
    return handleResponse(response);
  },

  // Get database details
  getDatabase: async (databaseId) => {
    const response = await fetch(`${API_BASE_URL}/databases/${databaseId}`, {
      method: 'GET',
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },

  // Update database configuration
  updateDatabase: async (databaseId, updateData) => {
    const response = await fetch(`${API_BASE_URL}/databases/${databaseId}`, {
      method: 'PUT',
      headers: getAuthHeaders(),
      body: JSON.stringify(updateData),
    });
    return handleResponse(response);
  },

  // Delete a database
  deleteDatabase: async (databaseId) => {
    const response = await fetch(`${API_BASE_URL}/databases/${databaseId}`, {
      method: 'DELETE',
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },
};

// API Service for Vector Operations
export const vectorApi = {
  // Store a single vector
  storeVector: async (databaseId, vectorData) => {
    const response = await fetch(`${API_BASE_URL}/databases/${databaseId}/vectors`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify(vectorData),
    });
    return handleResponse(response);
  },

  // Store multiple vectors in batch
  storeVectorsBatch: async (databaseId, vectorsData, upsert = false) => {
    const response = await fetch(`${API_BASE_URL}/databases/${databaseId}/vectors/batch`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify({
        vectors: vectorsData,
        upsert,
      }),
    });
    return handleResponse(response);
  },

  // List vectors in a database with pagination
  listVectors: async (databaseId, limit = 50, offset = 0) => {
    const response = await fetch(`${API_BASE_URL}/databases/${databaseId}/vectors?limit=${limit}&offset=${offset}`, {
      method: 'GET',
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },

  // Get a vector by ID
  getVector: async (databaseId, vectorId) => {
    const response = await fetch(`${API_BASE_URL}/databases/${databaseId}/vectors/${vectorId}`, {
      method: 'GET',
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },

  // Update a vector
  updateVector: async (databaseId, vectorId, vectorData) => {
    const response = await fetch(`${API_BASE_URL}/databases/${databaseId}/vectors/${vectorId}`, {
      method: 'PUT',
      headers: getAuthHeaders(),
      body: JSON.stringify(vectorData),
    });
    return handleResponse(response);
  },

  // Delete a vector
  deleteVector: async (databaseId, vectorId) => {
    const response = await fetch(`${API_BASE_URL}/databases/${databaseId}/vectors/${vectorId}`, {
      method: 'DELETE',
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },
};

// API Service for Search Operations
export const searchApi = {
  // Perform similarity search
  similaritySearch: async (databaseId, searchRequest) => {
    const response = await fetch(`${API_BASE_URL}/databases/${databaseId}/search`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify(searchRequest),
    });
    return handleResponse(response);
  },

  // Perform advanced similarity search with filters
  advancedSearch: async (databaseId, advancedSearchRequest) => {
    const response = await fetch(`${API_BASE_URL}/databases/${databaseId}/search/advanced`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify(advancedSearchRequest),
    });
    return handleResponse(response);
  },
};

// API Service for Index Management
export const indexApi = {
  // Create a new index
  createIndex: async (databaseId, indexData) => {
    const response = await fetch(`${API_BASE_URL}/databases/${databaseId}/indexes`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify(indexData),
    });
    return handleResponse(response);
  },

  // List all indexes in a database
  listIndexes: async (databaseId) => {
    const response = await fetch(`${API_BASE_URL}/databases/${databaseId}/indexes`, {
      method: 'GET',
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },

  // Get index details
  getIndex: async (databaseId, indexId) => {
    const response = await fetch(`${API_BASE_URL}/databases/${databaseId}/indexes/${indexId}`, {
      method: 'GET',
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },

  // Update index configuration
  updateIndex: async (databaseId, indexId, updateData) => {
    const response = await fetch(`${API_BASE_URL}/databases/${databaseId}/indexes/${indexId}`, {
      method: 'PUT',
      headers: getAuthHeaders(),
      body: JSON.stringify(updateData),
    });
    return handleResponse(response);
  },

  // Delete an index
  deleteIndex: async (databaseId, indexId) => {
    const response = await fetch(`${API_BASE_URL}/databases/${databaseId}/indexes/${indexId}`, {
      method: 'DELETE',
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },
};

// API Service for System Monitoring
export const monitoringApi = {
  // Health check
  healthCheck: async () => {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },

  // System status
  systemStatus: async () => {
    const response = await fetch(`${API_BASE_URL}/status`, {
      method: 'GET',
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },

  // Database-specific status
  databaseStatus: async (databaseId) => {
    const response = await fetch(`${API_BASE_URL}/databases/${databaseId}/status`, {
      method: 'GET',
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },
};

// API Service for Embedding Generation (placeholder for future implementation)
export const embeddingApi = {
  generateEmbedding: async (embeddingRequest) => {
    const response = await fetch(`${API_BASE_URL}/embeddings/generate`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify(embeddingRequest),
    });
    return handleResponse(response);
  },
};

// API Service for Lifecycle Management (placeholder for future implementation)
export const lifecycleApi = {
  configureRetention: async (databaseId, retentionConfig) => {
    const response = await fetch(`${API_BASE_URL}/databases/${databaseId}/lifecycle`, {
      method: 'PUT',
      headers: getAuthHeaders(),
      body: JSON.stringify(retentionConfig),
    });
    return handleResponse(response);
  },

  lifecycleStatus: async (databaseId) => {
    const response = await fetch(`${API_BASE_URL}/databases/${databaseId}/lifecycle/status`, {
      method: 'GET',
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },
};

// API Service for User Management
export const userApi = {
  listUsers: async () => {
    const response = await fetch(`${API_BASE_URL}/users`, {
      method: 'GET',
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },
  createUser: async (userData) => {
    const response = await fetch(`${API_BASE_URL}/users`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify(userData),
    });
    return handleResponse(response);
  },
  updateUser: async (userId, updateData) => {
    const response = await fetch(`${API_BASE_URL}/users/${userId}`, {
      method: 'PUT',
      headers: getAuthHeaders(),
      body: JSON.stringify(updateData),
    });
    return handleResponse(response);
  },
  deleteUser: async (userId) => {
    const response = await fetch(`${API_BASE_URL}/users/${userId}`, {
      method: 'DELETE',
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },
};

// API Service for Security Monitoring (Audit Logs)
export const securityApi = {
  listAuditLogs: async (limit = 50, offset = 0) => {
    const response = await fetch(`${API_BASE_URL}/audit/logs?limit=${limit}&offset=${offset}`, {
      method: 'GET',
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },
};

// API Service for API Key Management
export const apiKeyApi = {
  listKeys: async () => {
    const response = await fetch(`${API_BASE_URL}/apikeys`, {
      method: 'GET',
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },
  createKey: async (keyData) => {
    const response = await fetch(`${API_BASE_URL}/apikeys`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify(keyData),
    });
    return handleResponse(response);
  },
  revokeKey: async (keyId) => {
    const response = await fetch(`${API_BASE_URL}/apikeys/${keyId}`, {
      method: 'DELETE',
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },
};

// API Service for Alerting
export const alertApi = {
  listAlerts: async () => {
    const response = await fetch(`${API_BASE_URL}/alerts`, {
      method: 'GET',
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },
  createAlert: async (alertData) => {
    const response = await fetch(`${API_BASE_URL}/alerts`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify(alertData),
    });
    return handleResponse(response);
  },
  acknowledgeAlert: async (alertId) => {
    const response = await fetch(`${API_BASE_URL}/alerts/${alertId}/acknowledge`, {
      method: 'PUT',
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },
};

// API Service for Cluster Management
export const clusterApi = {
  listNodes: async () => {
    const response = await fetch(`${API_BASE_URL}/cluster/nodes`, {
      method: 'GET',
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },
  getNodeStatus: async (nodeId) => {
    const response = await fetch(`${API_BASE_URL}/cluster/nodes/${nodeId}/status`, {
      method: 'GET',
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },
};

// API Service for Performance Metrics
export const performanceApi = {
  getMetrics: async () => {
    const response = await fetch(`${API_BASE_URL}/metrics/performance`, {
      method: 'GET',
      headers: getAuthHeaders(),
    });
    return handleResponse(response);
  },
};

// API Service for Authentication (T219 - Authentication handlers)
export const authApi = {
  // Register a new user
  register: async (username, password, email = '', roles = []) => {
    const response = await fetch(`${API_BASE_URL}/auth/register`, {
      method: 'POST',
      headers: DEFAULT_HEADERS,
      body: JSON.stringify({ username, password, email, roles }),
    });
    return handleResponse(response);
  },

  // Login with username and password
  login: async (username, password) => {
    const response = await fetch(`${API_BASE_URL}/auth/login`, {
      method: 'POST',
      headers: DEFAULT_HEADERS,
      body: JSON.stringify({ username, password }),
    });
    const data = await handleResponse(response);
    // Store the token in localStorage
    if (data.token) {
      localStorage.setItem('jadevectordb_auth_token', data.token);
      localStorage.setItem('jadevectordb_user_id', data.user_id);
      localStorage.setItem('jadevectordb_username', data.username);
    }
    return data;
  },

  // Logout
  logout: async () => {
    const token = localStorage.getItem('jadevectordb_auth_token');
    if (!token) {
      throw new Error('No auth token found');
    }

    const response = await fetch(`${API_BASE_URL}/auth/logout`, {
      method: 'POST',
      headers: {
        ...DEFAULT_HEADERS,
        'Authorization': `Bearer ${token}`,
      },
    });

    // Clear local storage regardless of response
    localStorage.removeItem('jadevectordb_auth_token');
    localStorage.removeItem('jadevectordb_user_id');
    localStorage.removeItem('jadevectordb_username');
    localStorage.removeItem('jadevectordb_api_key');

    return handleResponse(response);
  },

  // Forgot password (request reset)
  forgotPassword: async (username, email) => {
    const response = await fetch(`${API_BASE_URL}/auth/forgot-password`, {
      method: 'POST',
      headers: DEFAULT_HEADERS,
      body: JSON.stringify({ username, email }),
    });
    return handleResponse(response);
  },

  // Reset password with token
  resetPassword: async (user_id, reset_token, new_password) => {
    const response = await fetch(`${API_BASE_URL}/auth/reset-password`, {
      method: 'POST',
      headers: DEFAULT_HEADERS,
      body: JSON.stringify({ user_id, reset_token, new_password }),
    });
    return handleResponse(response);
  },

  // Check if user is authenticated
  isAuthenticated: () => {
    const token = localStorage.getItem('jadevectordb_auth_token');
    return !!token;
  },

  // Get current user info from localStorage
  getCurrentUser: () => {
    return {
      user_id: localStorage.getItem('jadevectordb_user_id'),
      username: localStorage.getItem('jadevectordb_username'),
      token: localStorage.getItem('jadevectordb_auth_token'),
    };
  },
};

// API Service for User Management (T220 - User management handlers)
// Updated to match backend endpoints
export const usersApi = {
  // Create a new user (admin function)
  createUser: async (username, password, email = '', roles = []) => {
    const token = localStorage.getItem('jadevectordb_auth_token');
    const response = await fetch(`${API_BASE_URL}/users`, {
      method: 'POST',
      headers: {
        ...DEFAULT_HEADERS,
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify({ username, password, email, roles }),
    });
    return handleResponse(response);
  },

  // List all users
  listUsers: async (limit = 100, offset = 0) => {
    const token = localStorage.getItem('jadevectordb_auth_token');
    const response = await fetch(`${API_BASE_URL}/users?limit=${limit}&offset=${offset}`, {
      method: 'GET',
      headers: {
        ...DEFAULT_HEADERS,
        'Authorization': `Bearer ${token}`,
      },
    });
    return handleResponse(response);
  },

  // Get user details
  getUser: async (userId) => {
    const token = localStorage.getItem('jadevectordb_auth_token');
    const response = await fetch(`${API_BASE_URL}/users/${userId}`, {
      method: 'GET',
      headers: {
        ...DEFAULT_HEADERS,
        'Authorization': `Bearer ${token}`,
      },
    });
    return handleResponse(response);
  },

  // Update user
  updateUser: async (userId, updateData) => {
    const token = localStorage.getItem('jadevectordb_auth_token');
    const response = await fetch(`${API_BASE_URL}/users/${userId}`, {
      method: 'PUT',
      headers: {
        ...DEFAULT_HEADERS,
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify(updateData),
    });
    return handleResponse(response);
  },

  // Delete user
  deleteUser: async (userId) => {
    const token = localStorage.getItem('jadevectordb_auth_token');
    const response = await fetch(`${API_BASE_URL}/users/${userId}`, {
      method: 'DELETE',
      headers: {
        ...DEFAULT_HEADERS,
        'Authorization': `Bearer ${token}`,
      },
    });
    return handleResponse(response);
  },
};

// API Service for API Key Management (T221 - API key management)
// Updated to match backend endpoints
export const apiKeysApi = {
  // Create a new API key
  createApiKey: async (user_id, permissions = [], description = '', validity_days = 30) => {
    const token = localStorage.getItem('jadevectordb_auth_token');
    const response = await fetch(`${API_BASE_URL}/api-keys`, {
      method: 'POST',
      headers: {
        ...DEFAULT_HEADERS,
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify({ user_id, permissions, description, validity_days }),
    });
    const data = await handleResponse(response);
    // Optionally store the API key
    if (data.api_key) {
      localStorage.setItem('jadevectordb_api_key', data.api_key);
    }
    return data;
  },

  // List API keys
  listApiKeys: async (user_id = null) => {
    const token = localStorage.getItem('jadevectordb_auth_token');
    const url = user_id
      ? `${API_BASE_URL}/api-keys?user_id=${user_id}`
      : `${API_BASE_URL}/api-keys`;

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        ...DEFAULT_HEADERS,
        'Authorization': `Bearer ${token}`,
      },
    });
    return handleResponse(response);
  },

  // Revoke an API key
  revokeApiKey: async (key_id) => {
    const token = localStorage.getItem('jadevectordb_auth_token');
    const response = await fetch(`${API_BASE_URL}/api-keys/${key_id}`, {
      method: 'DELETE',
      headers: {
        ...DEFAULT_HEADERS,
        'Authorization': `Bearer ${token}`,
      },
    });
    return handleResponse(response);
  },
};