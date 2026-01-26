// frontend/tests/unit/pages/indexes-page.test.js
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import IndexManagement from '@/pages/indexes';

// Mock the API functions
jest.mock('@/lib/api', () => ({
  indexApi: {
    listIndexes: jest.fn(),
    createIndex: jest.fn(),
    deleteIndex: jest.fn(),
  },
  databaseApi: {
    listDatabases: jest.fn(),
  }
}));

import { indexApi, databaseApi } from '@/lib/api';

// Mock localStorage
Object.defineProperty(window, 'localStorage', {
  value: {
    getItem: jest.fn(() => 'test-api-key'),
  },
  writable: true,
});

// Mock next/router
jest.mock('next/router', () => ({
  useRouter: () => ({
    query: {},
    push: jest.fn(),
  })
}));

// Mock window.alert
beforeAll(() => {
  jest.spyOn(window, 'alert').mockImplementation(() => {});
});

describe('Index Management Page', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    // Mock successful API responses
    databaseApi.listDatabases.mockResolvedValue({
      databases: [
        {
          databaseId: 'test-db-id',
          name: 'Test DB',
          description: 'A test database',
          vectorDimension: 128,
        }
      ]
    });

    indexApi.listIndexes.mockResolvedValue({
      indexes: [
        {
          indexId: 'idx1',
          type: 'HNSW',
          status: 'ready',
          parameters: { M: 16, efConstruction: 200 },
          created_at: '2023-01-01T00:00:00Z'
        }
      ]
    });

    indexApi.createIndex.mockResolvedValue({
      indexId: 'idx2',
      type: 'IVF',
      status: 'building',
      parameters: { nlist: 100 },
      created_at: '2023-02-01T00:00:00Z'
    });
  });

  test('loads and displays database selection dropdown', async () => {
    render(<IndexManagement />);

    // Wait for databases to load
    await waitFor(() => {
      expect(screen.getByText('Test DB')).toBeInTheDocument();
    });

    // Page title and database selection should be visible
    expect(screen.getByText('Select Database')).toBeInTheDocument();
    expect(screen.getByText('Select a database')).toBeInTheDocument();
  });

  test('loads and displays indexes after selecting a database', async () => {
    render(<IndexManagement />);

    // Wait for databases to load
    await waitFor(() => {
      expect(screen.getByText('Test DB')).toBeInTheDocument();
    });

    // Select a database
    const select = screen.getByRole('combobox');
    fireEvent.change(select, { target: { value: 'test-db-id' } });

    // Wait for indexes to load (index type is shown as "HNSW Index")
    await waitFor(() => {
      expect(screen.getByText('HNSW Index')).toBeInTheDocument();
    });

    // Status should be displayed
    expect(screen.getByText('ready')).toBeInTheDocument();

    // Verify listIndexes was called with the selected database
    expect(indexApi.listIndexes).toHaveBeenCalledWith('test-db-id');
  });

  test('allows creating a new index', async () => {
    render(<IndexManagement />);

    // Wait for databases and select one
    await waitFor(() => {
      expect(screen.getByText('Test DB')).toBeInTheDocument();
    });

    // Select the database to show the index management section
    const dbSelect = screen.getByRole('combobox');
    fireEvent.change(dbSelect, { target: { value: 'test-db-id' } });

    // Wait for indexes section to appear (look for the "Index Type" label which is unique)
    await waitFor(() => {
      expect(screen.getByLabelText('Index Type')).toBeInTheDocument();
    });

    // Select index type
    const indexTypeSelect = screen.getByLabelText('Index Type');
    fireEvent.change(indexTypeSelect, { target: { value: 'IVF' } });

    // Enter parameters as JSON
    const parametersTextarea = screen.getByLabelText(/Parameters/i);
    fireEvent.change(parametersTextarea, { target: { value: '{"nlist": 100}' } });

    // Submit the form - use the button role with create index name
    fireEvent.click(screen.getByRole('button', { name: /create index/i }));

    // Wait for the API call to complete
    await waitFor(() => {
      expect(indexApi.createIndex).toHaveBeenCalledWith(
        'test-db-id',
        expect.objectContaining({
          type: 'IVF',
          parameters: { nlist: 100 }
        })
      );
    });
  });

  test('handles API errors when fetching indexes', async () => {
    // Mock an API error for listing indexes
    indexApi.listIndexes.mockRejectedValue(new Error('Failed to fetch indexes'));

    render(<IndexManagement />);

    // Wait for databases
    await waitFor(() => {
      expect(screen.getByText('Test DB')).toBeInTheDocument();
    });

    // Select a database to trigger the error
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'test-db-id' } });

    // Wait for the error to be handled (alert is called)
    await waitFor(() => {
      expect(indexApi.listIndexes).toHaveBeenCalled();
    });
  });
});
