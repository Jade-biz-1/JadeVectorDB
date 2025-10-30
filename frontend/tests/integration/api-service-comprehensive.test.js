// frontend/tests/integration/api-service-comprehensive.test.js
import { databaseApi, vectorApi, searchApi, indexApi, monitoringApi, embeddingApi, lifecycleApi } from '@/lib/api';

// Mock the fetch API for testing
global.fetch = jest.fn();

// Mock localStorage
Object.defineProperty(window, 'localStorage', {
  value: {
    getItem: jest.fn(() => 'test-api-key'),
  },
  writable: true,
});

describe('API Service Comprehensive Integration Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    global.fetch.mockClear();
  });

  describe('databaseApi comprehensive tests', () => {
    test('handles rate limiting responses', async () => {
      global.fetch.mockResolvedValue({
        ok: false,
        status: 429,
        json: () => Promise.resolve({ message: 'Too Many Requests' })
      });

      await expect(databaseApi.listDatabases()).rejects.toThrow('Too Many Requests');
    });

    test('handles network errors', async () => {
      global.fetch.mockRejectedValue(new TypeError('Network request failed'));

      await expect(databaseApi.listDatabases()).rejects.toThrow('Network error: Could not connect to the server. Please check your connection and try again.');
    });

    test('updateDatabase calls correct endpoint with PUT method', async () => {
      const updateData = { description: 'Updated description' };
      const mockResponse = { databaseId: '123', name: 'Test DB', description: 'Updated description' };
      global.fetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      const response = await databaseApi.updateDatabase('123', updateData);

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8080/v1/databases/123',
        expect.objectContaining({
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
            'X-API-Key': 'test-api-key'
          },
          body: JSON.stringify(updateData)
        })
      );
      expect(response).toEqual(mockResponse);
    });

    test('deleteDatabase calls correct endpoint with DELETE method', async () => {
      global.fetch.mockResolvedValue({
        ok: true,
        status: 204,
        json: () => Promise.resolve({})
      });

      const response = await databaseApi.deleteDatabase('123');

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8080/v1/databases/123',
        expect.objectContaining({
          method: 'DELETE',
          headers: {
            'Content-Type': 'application/json',
            'X-API-Key': 'test-api-key'
          }
        })
      );
      expect(response).toEqual({});
    });
  });

  describe('vectorApi comprehensive tests', () => {
    test('storeVectorsBatch calls correct endpoint with batch data', async () => {
      const vectorsData = [
        { id: 'vec1', values: [0.1, 0.2], metadata: { tag: 'example' } },
        { id: 'vec2', values: [0.3, 0.4], metadata: { tag: 'test' } }
      ];
      const mockResponse = { inserted: 2, updated: 0, errors: [] };
      global.fetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      const response = await vectorApi.storeVectorsBatch('db123', vectorsData, false);

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8080/v1/databases/db123/vectors/batch',
        expect.objectContaining({
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-API-Key': 'test-api-key'
          },
          body: JSON.stringify({
            vectors: vectorsData,
            upsert: false
          })
        })
      );
      expect(response).toEqual(mockResponse);
    });

    test('updateVector calls correct endpoint with vector data', async () => {
      const vectorData = { id: 'vec1', values: [0.5, 0.6], metadata: { tag: 'updated' } };
      const mockResponse = { ...vectorData };
      global.fetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      const response = await vectorApi.updateVector('db123', 'vec1', vectorData);

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8080/v1/databases/db123/vectors/vec1',
        expect.objectContaining({
          method: 'PUT',
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

  describe('searchApi comprehensive tests', () => {
    test('advancedSearch calls correct endpoint with filters', async () => {
      const searchRequest = {
        queryVector: [0.1, 0.2, 0.3],
        topK: 10,
        filters: { category: 'test' }
      };
      const mockResponse = { results: [], queryTimeMs: 5.2 };
      global.fetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      const response = await searchApi.advancedSearch('db123', searchRequest);

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8080/v1/databases/db123/search/advanced',
        expect.objectContaining({
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-API-Key': 'test-api-key'
          },
          body: JSON.stringify(searchRequest)
        })
      );
      expect(response).toEqual(mockResponse);
    });
  });

  describe('indexApi comprehensive tests', () => {
    test('updateIndex calls correct endpoint with update data', async () => {
      const updateData = { parameters: { M: 32 } };
      const mockResponse = { indexId: 'idx1', type: 'HNSW', parameters: { M: 32 } };
      global.fetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      const response = await indexApi.updateIndex('db123', 'idx1', updateData);

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8080/v1/databases/db123/indexes/idx1',
        expect.objectContaining({
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
            'X-API-Key': 'test-api-key'
          },
          body: JSON.stringify(updateData)
        })
      );
      expect(response).toEqual(mockResponse);
    });
  });

  describe('embeddingApi tests', () => {
    test('generateEmbedding calls correct endpoint', async () => {
      const embeddingRequest = { text: 'Hello World', model: 'default' };
      const mockResponse = { embedding: [0.1, 0.2, 0.3], model: 'default' };
      global.fetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      const response = await embeddingApi.generateEmbedding(embeddingRequest);

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8080/v1/embeddings/generate',
        expect.objectContaining({
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-API-Key': 'test-api-key'
          },
          body: JSON.stringify(embeddingRequest)
        })
      );
      expect(response).toEqual(mockResponse);
    });
  });

  describe('lifecycleApi tests', () => {
    test('configureRetention calls correct endpoint', async () => {
      const retentionConfig = { maxAgeDays: 30, archiveOnExpire: true };
      const mockResponse = { status: 'success' };
      global.fetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      const response = await lifecycleApi.configureRetention('db123', retentionConfig);

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8080/v1/databases/db123/lifecycle',
        expect.objectContaining({
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
            'X-API-Key': 'test-api-key'
          },
          body: JSON.stringify(retentionConfig)
        })
      );
      expect(response).toEqual(mockResponse);
    });

    test('lifecycleStatus calls correct endpoint', async () => {
      const mockResponse = { status: 'active', nextCleanup: '2023-12-01T00:00:00Z' };
      global.fetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      const response = await lifecycleApi.lifecycleStatus('db123');

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8080/v1/databases/db123/lifecycle/status',
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
});