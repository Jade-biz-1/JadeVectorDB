// tests/unit/hooks/useDatabases.test.js
// Unit tests for the useDatabases custom hook.

import { renderHook, waitFor, act } from '@testing-library/react';

jest.mock('@/lib/api', () => ({
  databaseApi: {
    listDatabases: jest.fn(),
  },
}));

const { databaseApi } = require('@/lib/api');
import { useDatabases } from '@/hooks/useDatabases';

const RAW_DATABASES = [
  { databaseId: 'db-1', name: 'AlphaDB' },
  { databaseId: 'db-2', name: 'BetaDB'  },
];

describe('useDatabases hook', () => {
  beforeEach(() => jest.clearAllMocks());

  it('returns an empty list and loading=true initially', () => {
    databaseApi.listDatabases.mockImplementation(() => new Promise(() => {}));
    const { result } = renderHook(() => useDatabases());
    expect(result.current.databases).toEqual([]);
    expect(result.current.loading).toBe(true);
    expect(result.current.error).toBe('');
  });

  it('normalises databases to { id, name } shape', async () => {
    databaseApi.listDatabases.mockResolvedValue({ databases: RAW_DATABASES });
    const { result } = renderHook(() => useDatabases());

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.databases).toEqual([
      { id: 'db-1', name: 'AlphaDB' },
      { id: 'db-2', name: 'BetaDB'  },
    ]);
  });

  it('sets error when the API call fails', async () => {
    databaseApi.listDatabases.mockRejectedValue(new Error('Network error'));
    const { result } = renderHook(() => useDatabases());

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.error).toMatch(/error fetching databases/i);
    expect(result.current.databases).toEqual([]);
  });

  it('handles an empty databases array gracefully', async () => {
    databaseApi.listDatabases.mockResolvedValue({ databases: [] });
    const { result } = renderHook(() => useDatabases());

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.databases).toEqual([]);
    expect(result.current.error).toBe('');
  });

  it('handles a missing databases field gracefully', async () => {
    databaseApi.listDatabases.mockResolvedValue({});
    const { result } = renderHook(() => useDatabases());

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.databases).toEqual([]);
  });

  it('refetches when refetch() is called', async () => {
    databaseApi.listDatabases
      .mockResolvedValueOnce({ databases: [RAW_DATABASES[0]] })
      .mockResolvedValueOnce({ databases: RAW_DATABASES });

    const { result } = renderHook(() => useDatabases());
    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.databases).toHaveLength(1);

    await act(async () => { result.current.refetch(); });
    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.databases).toHaveLength(2);
    expect(databaseApi.listDatabases).toHaveBeenCalledTimes(2);
  });

  it('calls listDatabases exactly once on mount', async () => {
    databaseApi.listDatabases.mockResolvedValue({ databases: [] });
    renderHook(() => useDatabases());

    await waitFor(() => expect(databaseApi.listDatabases).toHaveBeenCalledTimes(1));
  });
});
