// frontend/tests/unit/hooks/useApi.test.js
import { renderHook, act } from '@testing-library/react';
import { useApi } from '@/hooks/useApi';

// Mock the API functions
jest.mock('@/lib/api', () => ({
  databaseApi: {
    listDatabases: jest.fn(),
    createDatabase: jest.fn(),
  },
  searchApi: {
    similaritySearch: jest.fn(),
  }
}));

import { databaseApi, searchApi } from '@/lib/api';

// Mock localStorage
Object.defineProperty(window, 'localStorage', {
  value: {
    getItem: jest.fn(() => 'test-api-key'),
  },
  writable: true,
});

describe('useApi Hook', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('initializes with loading state', () => {
    const { result } = renderHook(() => useApi());
    
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
    expect(result.current.data).toBeNull();
  });

  test('executes API calls correctly', async () => {
    const mockDatabases = [
      { databaseId: '1', name: 'DB1', vectorDimension: 128 },
      { databaseId: '2', name: 'DB2', vectorDimension: 256 }
    ];
    
    databaseApi.listDatabases.mockResolvedValue({ databases: mockDatabases });
    
    const { result } = renderHook(() => useApi());
    
    await act(async () => {
      const response = await result.current.execute(databaseApi.listDatabases);
      expect(response.databases).toEqual(mockDatabases);
    });
    
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
    // Note: The data property would depend on the actual implementation of useApi
  });

  test('handles API errors correctly', async () => {
    const error = new Error('API Error');
    databaseApi.listDatabases.mockRejectedValue(error);
    
    const { result } = renderHook(() => useApi());
    
    await act(async () => {
      try {
        await result.current.execute(databaseApi.listDatabases);
      } catch (err) {
        // Error is handled in the hook
      }
    });
    
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toEqual(error);
  });

  test('clears error when executing new request', async () => {
    const error = new Error('API Error');
    databaseApi.listDatabases.mockRejectedValue(error);
    
    const { result } = renderHook(() => useApi());
    
    // First, trigger an error
    await act(async () => {
      try {
        await result.current.execute(databaseApi.listDatabases);
      } catch (err) {
        // Error is handled in the hook
      }
    });
    
    expect(result.current.error).toEqual(error);
    
    // Then, mock a successful response
    databaseApi.listDatabases.mockResolvedValue({ databases: [] });
    
    // Execute another request
    await act(async () => {
      await result.current.execute(databaseApi.listDatabases);
    });
    
    expect(result.current.error).toBeNull();
  });

  test('loading state updates correctly', async () => {
    // Mock a slow API call
    const promise = Promise.resolve({ databases: [] });
    databaseApi.listDatabases.mockReturnValue(promise);
    
    const { result } = renderHook(() => useApi());
    
    // Start the API call
    const callPromise = act(async () => {
      await result.current.execute(databaseApi.listDatabases);
    });
    
    // Check that loading is true during the call
    expect(result.current.loading).toBe(true);
    
    // Wait for the call to complete
    await callPromise;
    
    // Check that loading is false after the call
    expect(result.current.loading).toBe(false);
  });
});