const axios = require('axios');

// Helper function to create axios instance with auth
const createApiClient = (baseUrl, apiKey) => {
  const headers = {
    'Content-Type': 'application/json'
  };

  if (apiKey) {
    headers['Authorization'] = `Bearer ${apiKey}`;
  }

  return axios.create({
    baseURL: baseUrl,
    timeout: 30000, // 30 seconds timeout
    headers: headers
  });
};

// Database operations
const createDatabase = async (baseUrl, apiKey, name, description = '', dimension = 128, indexType = 'HNSW') => {
  const client = createApiClient(baseUrl, apiKey);
  
  const payload = {
    name,
    description,
    vectorDimension: dimension,
    indexType
  };
  
  try {
    const response = await client.post('/v1/databases', payload);
    return response.data;
  } catch (error) {
    throw new Error(`Failed to create database: ${error.response?.data?.message || error.message}`);
  }
};

const listDatabases = async (baseUrl, apiKey) => {
  const client = createApiClient(baseUrl, apiKey);
  
  try {
    const response = await client.get('/v1/databases');
    return response.data;
  } catch (error) {
    throw new Error(`Failed to list databases: ${error.response?.data?.message || error.message}`);
  }
};

const getDatabase = async (baseUrl, apiKey, databaseId) => {
  const client = createApiClient(baseUrl, apiKey);
  
  try {
    const response = await client.get(`/v1/databases/${databaseId}`);
    return response.data;
  } catch (error) {
    throw new Error(`Failed to get database: ${error.response?.data?.message || error.message}`);
  }
};

// Vector operations
const storeVector = async (baseUrl, apiKey, databaseId, vectorId, values, metadata = null) => {
  const client = createApiClient(baseUrl, apiKey);
  
  const payload = {
    id: vectorId,
    values
  };
  
  if (metadata) {
    payload.metadata = metadata;
  }
  
  try {
    const response = await client.post(`/v1/databases/${databaseId}/vectors`, payload);
    return response.data;
  } catch (error) {
    throw new Error(`Failed to store vector: ${error.response?.data?.message || error.message}`);
  }
};

const retrieveVector = async (baseUrl, apiKey, databaseId, vectorId) => {
  const client = createApiClient(baseUrl, apiKey);
  
  try {
    const response = await client.get(`/v1/databases/${databaseId}/vectors/${vectorId}`);
    return response.data;
  } catch (error) {
    if (error.response?.status === 404) {
      return null; // Vector not found
    }
    throw new Error(`Failed to retrieve vector: ${error.response?.data?.message || error.message}`);
  }
};

const deleteVector = async (baseUrl, apiKey, databaseId, vectorId) => {
  const client = createApiClient(baseUrl, apiKey);
  
  try {
    const response = await client.delete(`/v1/databases/${databaseId}/vectors/${vectorId}`);
    return response.data;
  } catch (error) {
    throw new Error(`Failed to delete vector: ${error.response?.data?.message || error.message}`);
  }
};

// Search operations
const searchVectors = async (baseUrl, apiKey, databaseId, queryVector, topK = 10, threshold = null) => {
  const client = createApiClient(baseUrl, apiKey);
  
  const payload = {
    queryVector,
    topK
  };
  
  if (threshold !== null) {
    payload.threshold = threshold;
  }
  
  try {
    const response = await client.post(`/v1/databases/${databaseId}/search`, payload);
    return response.data;
  } catch (error) {
    throw new Error(`Failed to search vectors: ${error.response?.data?.message || error.message}`);
  }
};

// System operations
const getHealth = async (baseUrl, apiKey) => {
  const client = createApiClient(baseUrl, apiKey);
  
  try {
    const response = await client.get('/health');
    return response.data;
  } catch (error) {
    throw new Error(`Failed to get health status: ${error.response?.data?.message || error.message}`);
  }
};

const getStatus = async (baseUrl, apiKey) => {
  const client = createApiClient(baseUrl, apiKey);
  
  try {
    const response = await client.get('/status');
    return response.data;
  } catch (error) {
    throw new Error(`Failed to get system status: ${error.response?.data?.message || error.message}`);
  }
};

// User management operations
const createUser = async (baseUrl, apiKey, email, role, password = null) => {
  const client = createApiClient(baseUrl, apiKey);

  const payload = {
    email,
    role
  };

  if (password) {
    payload.password = password;
  }

  try {
    const response = await client.post('/api/v1/users', payload);
    return response.data;
  } catch (error) {
    throw new Error(`Failed to create user: ${error.response?.data?.message || error.message}`);
  }
};

const listUsers = async (baseUrl, apiKey, role = null, status = null) => {
  const client = createApiClient(baseUrl, apiKey);

  const params = {};
  if (role) params.role = role;
  if (status) params.status = status;

  try {
    const response = await client.get('/api/v1/users', { params });
    return response.data;
  } catch (error) {
    throw new Error(`Failed to list users: ${error.response?.data?.message || error.message}`);
  }
};

const getUser = async (baseUrl, apiKey, email) => {
  const client = createApiClient(baseUrl, apiKey);

  try {
    const response = await client.get(`/api/v1/users/${email}`);
    return response.data;
  } catch (error) {
    if (error.response?.status === 404) {
      throw new Error(`User not found: ${email}`);
    }
    throw new Error(`Failed to get user: ${error.response?.data?.message || error.message}`);
  }
};

const updateUser = async (baseUrl, apiKey, email, role = null, status = null) => {
  const client = createApiClient(baseUrl, apiKey);

  const payload = {};
  if (role) payload.role = role;
  if (status) payload.status = status;

  if (Object.keys(payload).length === 0) {
    throw new Error('At least one of role or status must be provided');
  }

  try {
    const response = await client.put(`/api/v1/users/${email}`, payload);
    return response.data;
  } catch (error) {
    throw new Error(`Failed to update user: ${error.response?.data?.message || error.message}`);
  }
};

const deleteUser = async (baseUrl, apiKey, email) => {
  const client = createApiClient(baseUrl, apiKey);

  try {
    const response = await client.delete(`/api/v1/users/${email}`);
    return response.data;
  } catch (error) {
    throw new Error(`Failed to delete user: ${error.response?.data?.message || error.message}`);
  }
};

const activateUser = async (baseUrl, apiKey, email) => {
  return updateUser(baseUrl, apiKey, email, null, 'active');
};

const deactivateUser = async (baseUrl, apiKey, email) => {
  return updateUser(baseUrl, apiKey, email, null, 'inactive');
};

module.exports = {
  createDatabase,
  listDatabases,
  getDatabase,
  storeVector,
  retrieveVector,
  deleteVector,
  searchVectors,
  getHealth,
  getStatus,
  createUser,
  listUsers,
  getUser,
  updateUser,
  deleteUser,
  activateUser,
  deactivateUser
};