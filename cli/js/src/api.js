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

module.exports = {
  createDatabase,
  listDatabases,
  getDatabase,
  storeVector,
  retrieveVector,
  deleteVector,
  searchVectors,
  getHealth,
  getStatus
};