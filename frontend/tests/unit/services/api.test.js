// frontend/tests/unit/services/api.test.js
import { databaseApi, searchApi, vectorApi, indexApi, monitoringApi, embeddingApi, lifecycleApi, securityApi } from '@/lib/api';

// Mock the fetch API for testing
global.fetch = jest.fn();

// Mock localStorage - the actual implementation uses 'jadevectordb_auth_token'
const mockLocalStorage = {
  getItem: jest.fn((key) => {
    if (key === 'jadevectordb_auth_token') return 'test-auth-token';
    return null;
  }),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};

Object.defineProperty(window, 'localStorage', {
  value: mockLocalStorage,
  writable: true,
});

// Suppress console.warn for "No auth token found" messages
const originalWarn = console.warn;
beforeAll(() => {
  console.warn = jest.fn();
});
afterAll(() => {
  console.warn = originalWarn;
});

describe('API Service Unit Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    global.fetch.mockClear();
    // Reset localStorage mock to return token
    mockLocalStorage.getItem.mockImplementation((key) => {
      if (key === 'jadevectordb_auth_token') return 'test-auth-token';
      return null;
    });
  });

  describe('handleResponse function', () => {
    test('returns JSON response when status is OK', async () => {
      const mockResponse = { data: 'test' };
      global.fetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      const response = await databaseApi.listDatabases();

      expect(response).toEqual(mockResponse);
    });

    test('throws error when response is not OK', async () => {
      global.fetch.mockResolvedValue({
        ok: false,
        status: 400,
        json: () => Promise.resolve({ message: 'Bad Request' })
      });

      await expect(databaseApi.listDatabases()).rejects.toThrow('Bad Request');
    });

    test('throws generic error when response is not OK and no message is provided', async () => {
      global.fetch.mockResolvedValue({
        ok: false,
        status: 500,
        json: () => Promise.resolve({})
      });

      await expect(databaseApi.listDatabases()).rejects.toThrow('API error: 500');
    });
  });

  describe('databaseApi', () => {
    test('listDatabases calls correct endpoint with query params', async () => {
      const mockResponse = { databases: [] };
      global.fetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      const response = await databaseApi.listDatabases(10, 20);

      // Actual implementation uses /api/... URLs (proxied by Next.js)
      expect(global.fetch).toHaveBeenCalledWith(
        '/api/databases?limit=10&offset=20',
        expect.objectContaining({
          method: 'GET',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test-auth-token'
          })
        })
      );
      expect(response).toEqual(mockResponse);
    });

    test('createDatabase calls correct endpoint with POST method', async () => {
      const newDatabase = { name: 'Test DB', vectorDimension: 128 };
      const mockResponse = { databaseId: '123', ...newDatabase };
      global.fetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      const response = await databaseApi.createDatabase(newDatabase);

      expect(global.fetch).toHaveBeenCalledWith(
        '/api/databases',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test-auth-token'
          }),
          body: JSON.stringify(newDatabase)
        })
      );
      expect(response).toEqual(mockResponse);
    });

    test('getDatabase calls correct endpoint with database ID', async () => {
      const mockResponse = { databaseId: '123', name: 'Test DB' };
      global.fetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      const response = await databaseApi.getDatabase('123');

      expect(global.fetch).toHaveBeenCalledWith(
        '/api/databases/123',
        expect.objectContaining({
          method: 'GET',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test-auth-token'
          })
        })
      );
      expect(response).toEqual(mockResponse);
    });
  });

  describe('vectorApi', () => {
    test('storeVector calls correct endpoint with vector data', async () => {
      const vectorData = { id: 'vec1', values: [0.1, 0.2, 0.3], metadata: { tag: 'example' } };
      const mockResponse = { status: 'success' };
      global.fetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      const response = await vectorApi.storeVector('db123', vectorData);

      expect(global.fetch).toHaveBeenCalledWith(
        '/api/databases/db123/vectors',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test-auth-token'
          }),
          body: JSON.stringify(vectorData)
        })
      );
      expect(response).toEqual(mockResponse);
    });
  });

  describe('indexApi', () => {
    test('createIndex calls correct endpoint with index data', async () => {
      const indexData = { type: 'HNSW', parameters: { M: 16 } };
      const mockResponse = { indexId: 'idx1', ...indexData };
      global.fetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      const response = await indexApi.createIndex('db123', indexData);

      expect(global.fetch).toHaveBeenCalledWith(
        '/api/databases/db123/indexes',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test-auth-token'
          }),
          body: JSON.stringify(indexData)
        })
      );
      expect(response).toEqual(mockResponse);
    });
  });

  describe('monitoringApi', () => {
    test('healthCheck calls health endpoint', async () => {
      const mockResponse = { status: 'healthy' };
      global.fetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      const response = await monitoringApi.healthCheck();

      expect(global.fetch).toHaveBeenCalledWith(
        '/api/health',
        expect.objectContaining({
          method: 'GET',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test-auth-token'
          })
        })
      );
      expect(response).toEqual(mockResponse);
    });

    test('systemStatus calls status endpoint', async () => {
      const mockResponse = { status: 'operational' };
      global.fetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      const response = await monitoringApi.systemStatus();

      expect(global.fetch).toHaveBeenCalledWith(
        '/api/status',
        expect.objectContaining({
          method: 'GET',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test-auth-token'
          })
        })
      );
      expect(response).toEqual(mockResponse);
    });
  });

  describe('securityApi', () => {
    test('listAuditLogs calls correct endpoint with limit param', async () => {
      const mockResponse = { events: [] };
      global.fetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      const response = await securityApi.listAuditLogs(10);

      expect(global.fetch).toHaveBeenCalledWith(
        '/api/security/audit-log?limit=10',
        expect.objectContaining({
          method: 'GET',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test-auth-token'
          })
        })
      );
      expect(response).toEqual(mockResponse);
    });

    test('listAuditLogs uses default limit of 50', async () => {
      const mockResponse = { events: [] };
      global.fetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      await securityApi.listAuditLogs();

      expect(global.fetch).toHaveBeenCalledWith(
        '/api/security/audit-log?limit=50',
        expect.objectContaining({
          method: 'GET'
        })
      );
    });
  });

  describe('getAuthHeaders function', () => {
    test('returns headers with Authorization Bearer token when available', async () => {
      const mockResponse = { databases: [] };
      global.fetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      await databaseApi.listDatabases();

      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          headers: expect.objectContaining({
            'Authorization': 'Bearer test-auth-token'
          })
        })
      );
    });

    test('returns default headers when no auth token is available', async () => {
      // Mock localStorage to return null for auth token
      mockLocalStorage.getItem.mockImplementation(() => null);

      const mockResponse = { databases: [] };
      global.fetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      await databaseApi.listDatabases();

      // When no token, should only have Content-Type header
      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          headers: {
            'Content-Type': 'application/json'
          }
        })
      );
    });
  });
});
