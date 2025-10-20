// frontend/tests/integration/api-service.test.js
import { databaseApi, searchApi } from '@@/services/api';

// Mock the fetch API for testing
global.fetch = jest.fn();

describe('API Service Integration', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    
    // Mock localStorage
    Object.defineProperty(window, 'localStorage', {
      value: {
        getItem: jest.fn(() => 'test-api-key'),
      },
      writable: true,
    });
  });

  test('databaseApi.listDatabases calls the correct endpoint', async () => {
    const mockResponse = {
      databases: [
        { databaseId: 'db1', name: 'Test DB', vectorDimension: 128 }
      ],
      total: 1
    };
    
    global.fetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockResponse)
    });

    const response = await databaseApi.listDatabases();

    expect(global.fetch).toHaveBeenCalledWith(
      'http://localhost:8080/v1/databases?limit=20&offset=0', 
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': 'test-api-key'
        }
      }
    );
    
    expect(response).toEqual(mockResponse);
  });

  test('databaseApi.createDatabase calls the correct endpoint', async () => {
    const newDatabase = { name: 'New DB', vectorDimension: 256 };
    const mockResponse = { databaseId: 'new-db-id', ...newDatabase };
    
    global.fetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockResponse)
    });

    const response = await databaseApi.createDatabase(newDatabase);

    expect(global.fetch).toHaveBeenCalledWith(
      'http://localhost:8080/v1/databases',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': 'test-api-key'
        },
        body: JSON.stringify(newDatabase)
      }
    );
    
    expect(response).toEqual(mockResponse);
  });

  test('searchApi.similaritySearch calls the correct endpoint', async () => {
    const searchRequest = {
      queryVector: [0.1, 0.2, 0.3],
      topK: 10
    };
    const mockResponse = {
      results: [
        { id: 'result1', similarity: 0.95 }
      ]
    };
    
    global.fetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockResponse)
    });

    const response = await searchApi.similaritySearch('test-db-id', searchRequest);

    expect(global.fetch).toHaveBeenCalledWith(
      'http://localhost:8080/v1/databases/test-db-id/search',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': 'test-api-key'
        },
        body: JSON.stringify(searchRequest)
      }
    );
    
    expect(response).toEqual(mockResponse);
  });

  test('handles API errors correctly', async () => {
    global.fetch.mockResolvedValue({
      ok: false,
      json: () => Promise.resolve({ message: 'Invalid request' }),
      status: 400
    });

    await expect(databaseApi.listDatabases()).rejects.toThrow('Invalid request');
  });

  test('handles network errors correctly', async () => {
    global.fetch.mockRejectedValue(new Error('Network error'));

    await expect(databaseApi.listDatabases()).rejects.toThrow('Network error');
  });
});