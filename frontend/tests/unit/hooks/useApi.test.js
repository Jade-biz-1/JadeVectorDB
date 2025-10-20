// frontend/tests/unit/hooks/useApi.test.js
import { renderHook, waitFor } from '@testing-library/react';
import useApi from '@@/hooks/useApi';

// Mock API function for testing
const mockApiFunction = jest.fn();

describe('useApi Hook', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('initially sets loading to true and data to null', () => {
    mockApiFunction.mockResolvedValue('test data');
    
    const { result } = renderHook(() => useApi(mockApiFunction));
    
    expect(result.current.loading).toBe(true);
    expect(result.current.data).toBeNull();
    expect(result.current.error).toBeNull();
  });

  test('sets data and loading state when API call succeeds', async () => {
    const testData = { id: 1, name: 'Test' };
    mockApiFunction.mockResolvedValue(testData);
    
    const { result } = renderHook(() => useApi(mockApiFunction));
    
    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });
    
    expect(result.current.data).toEqual(testData);
    expect(result.current.error).toBeNull();
  });

  test('sets error when API call fails', async () => {
    const error = new Error('API Error');
    mockApiFunction.mockRejectedValue(error);
    
    const { result } = renderHook(() => useApi(mockApiFunction));
    
    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });
    
    expect(result.current.data).toBeNull();
    expect(result.current.error).toBe('API Error');
  });

  test('refetch function works correctly', async () => {
    const testData = { id: 2, name: 'Refetched' };
    mockApiFunction
      .mockResolvedValueOnce('initial data') // First call
      .mockResolvedValueOnce(testData); // After refetch
    
    const { result } = renderHook(() => useApi(mockApiFunction));
    
    // Wait for initial call to complete
    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });
    
    // Call refetch
    result.current.refetch();
    
    // Wait for refetch to complete
    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });
    
    expect(result.current.data).toEqual(testData);
  });

  test('only calls API function once initially', () => {
    mockApiFunction.mockResolvedValue('test data');
    
    renderHook(() => useApi(mockApiFunction));
    
    expect(mockApiFunction).toHaveBeenCalledTimes(1);
  });
});