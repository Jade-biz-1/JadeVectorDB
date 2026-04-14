// Shared mock for src/lib/api.js
const databaseApi = {
  listDatabases: jest.fn(),
  createDatabase: jest.fn(),
  getDatabase: jest.fn(),
  updateDatabase: jest.fn(),
  deleteDatabase: jest.fn(),
};

const vectorApi = {
  storeVector: jest.fn(),
  storeVectorsBatch: jest.fn(),
  listVectors: jest.fn(),
  getVector: jest.fn(),
  updateVector: jest.fn(),
  deleteVector: jest.fn(),
};

const searchApi = {
  similaritySearch: jest.fn(),
};

const authApi = {
  login: jest.fn(),
  register: jest.fn(),
  logout: jest.fn(),
  changePassword: jest.fn(),
  forgotPassword: jest.fn(),
  resetPassword: jest.fn(),
};

const usersApi = {
  listUsers: jest.fn(),
  createUser: jest.fn(),
  updateUser: jest.fn(),
  deleteUser: jest.fn(),
  resetPassword: jest.fn(),
};

const apiKeysApi = {
  listApiKeys: jest.fn(),
  createApiKey: jest.fn(),
  deleteApiKey: jest.fn(),
};

const monitoringApi = {
  getSystemStatus: jest.fn(),
  getMetrics: jest.fn(),
};

const clusterApi = {
  listNodes: jest.fn(),
  getNodeStatus: jest.fn(),
};

const securityApi = {
  listAuditLogs: jest.fn(),
};

const alertApi = {
  listAlerts: jest.fn(),
  createAlert: jest.fn(),
  deleteAlert: jest.fn(),
};

const performanceApi = {
  getMetrics: jest.fn(),
};

module.exports = {
  databaseApi,
  vectorApi,
  searchApi,
  authApi,
  usersApi,
  apiKeysApi,
  monitoringApi,
  clusterApi,
  securityApi,
  alertApi,
  performanceApi,
};
