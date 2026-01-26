// frontend/tests/unit/hooks/useApi.test.js
import { renderHook, waitFor, act } from '@testing-library/react';
import useApi from '@/hooks/useApi';  // Default export

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

// Suppress console.error for expected API errors
const originalError = console.error;
beforeAll(() => {
  console.error = jest.fn();
});
afterAll(() => {
  console.error = originalError;
});

describe('useApi Hook', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('initializes with loading state and fetches data', async () => {
    const mockDatabases = [
      { databaseId: '1', name: 'DB1' },
      { databaseId: '2', name: 'DB2' }
    ];

    databaseApi.listDatabases.mockResolvedValue({ databases: mockDatabases });

    // useApi takes an API function and dependencies array
    const { result } = renderHook(() =>
      useApi(() => databaseApi.listDatabases(), [])
    );

    // Initially loading
    expect(result.current.loading).toBe(true);

    // Wait for data to load
    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.data).toEqual({ databases: mockDatabases });
    expect(result.current.error).toBeNull();
  });

  test('handles API errors correctly', async () => {
    const errorMessage = 'Network error';
    databaseApi.listDatabases.mockRejectedValue(new Error(errorMessage));

    const { result } = renderHook(() =>
      useApi(() => databaseApi.listDatabases(), [])
    );

    // Wait for error to be set
    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.error).toBe(errorMessage);
    expect(result.current.data).toBeNull();
  });

  test('refetch function re-fetches data', async () => {
    const initialData = { databases: [{ id: '1' }] };
    const updatedData = { databases: [{ id: '1' }, { id: '2' }] };

    databaseApi.listDatabases
      .mockResolvedValueOnce(initialData)
      .mockResolvedValueOnce(updatedData);

    const { result } = renderHook(() =>
      useApi(() => databaseApi.listDatabases(), [])
    );

    // Wait for initial data
    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.data).toEqual(initialData);

    // Call refetch
    await act(async () => {
      await result.current.refetch();
    });

    expect(result.current.data).toEqual(updatedData);
  });

  test('returns loading, error, data, and refetch', async () => {
    databaseApi.listDatabases.mockResolvedValue({ databases: [] });

    const { result } = renderHook(() =>
      useApi(() => databaseApi.listDatabases(), [])
    );

    // Check that all expected properties exist
    expect(result.current).toHaveProperty('loading');
    expect(result.current).toHaveProperty('error');
    expect(result.current).toHaveProperty('data');
    expect(result.current).toHaveProperty('refetch');
    expect(typeof result.current.refetch).toBe('function');
  });

  test('handles different API functions', async () => {
    const searchResults = { results: [{ id: 'vec1', score: 0.95 }] };
    searchApi.similaritySearch.mockResolvedValue(searchResults);

    const { result } = renderHook(() =>
      useApi(() => searchApi.similaritySearch(), [])
    );

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.data).toEqual(searchResults);
  });

  test('handles error message fallback', async () => {
    // Error without message property
    databaseApi.listDatabases.mockRejectedValue({});

    const { result } = renderHook(() =>
      useApi(() => databaseApi.listDatabases(), [])
    );

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.error).toBe('An error occurred');
  });

  test('sets loading to true during refetch', async () => {
    let resolvePromise;
    const promise = new Promise((resolve) => {
      resolvePromise = resolve;
    });

    databaseApi.listDatabases
      .mockResolvedValueOnce({ databases: [] })
      .mockReturnValueOnce(promise);

    const { result } = renderHook(() =>
      useApi(() => databaseApi.listDatabases(), [])
    );

    // Wait for initial load
    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    // Start refetch (don't await)
    act(() => {
      result.current.refetch();
    });

    // Should be loading during refetch
    expect(result.current.loading).toBe(true);

    // Resolve the promise
    await act(async () => {
      resolvePromise({ databases: [{ id: '1' }] });
    });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });
  });
});
