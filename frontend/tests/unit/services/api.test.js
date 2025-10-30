// frontend/tests/unit/services/api.test.js
import { databaseApi, searchApi, vectorApi, indexApi, monitoringApi, embeddingApi, lifecycleApi } from '@/lib/api';

// Mock the fetch API for testing
global.fetch = jest.fn();

// Mock localStorage
Object.defineProperty(window, 'localStorage', {
  value: {
    getItem: jest.fn(() => 'test-api-key'),
  },
  writable: true,
});

describe('API Service Unit Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    global.fetch.mockClear();
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

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8080/v1/databases?limit=10&offset=20',
        expect.objectContaining({
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
            'X-API-Key': 'test-api-key'
          }
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
        'http://localhost:8080/v1/databases',
        expect.objectContaining({
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-API-Key': 'test-api-key'
          },
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
        'http://localhost:8080/v1/databases/123',
        expect.objectContaining({
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
            'X-API-Key': 'test-api-key'
          }
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
        'http://localhost:8080/v1/databases/db123/vectors',
        expect.objectContaining({
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-API-Key': 'test-api-key'
          },
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
        'http://localhost:8080/v1/databases/db123/indexes',
        expect.objectContaining({
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-API-Key': 'test-api-key'
          },
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
        'http://localhost:8080/v1/health',
        expect.objectContaining({
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
            'X-API-Key': 'test-api-key'
          }
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
        'http://localhost:8080/v1/status',
        expect.objectContaining({
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
            'X-API-Key': 'test-api-key'
          }
        })
      );
      expect(response).toEqual(mockResponse);
    });
  });

  describe('getAuthHeaders function', () => {
    test('returns headers with API key when available', () => {
      // This is covered implicitly in all the other tests
      // that verify the X-API-Key header is included
    });

    test('returns default headers when no API key is available', async () => {
      // Mock localStorage to return null
      Object.defineProperty(window, 'localStorage', {
        value: {
          getItem: jest.fn(() => null),
        },
        writable: true,
      });

      const mockResponse = { databases: [] };
      global.fetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      await databaseApi.listDatabases();

      expect(global.fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          headers: {
            'Content-Type': 'application/json'
            // No X-API-Key header should be present
          }
        })
      );
    });
  });
});