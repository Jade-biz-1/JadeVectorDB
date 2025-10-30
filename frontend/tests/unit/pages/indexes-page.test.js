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

// Mock router
jest.mock('next/router', () => ({
  useRouter: () => ({
    query: { databaseId: 'test-db-id' },
    push: jest.fn(),
  })
}));

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

  test('loads and displays existing indexes', async () => {
    render(<IndexManagement />);
    
    // Wait for indexes to load
    await waitFor(() => {
      expect(screen.getByText('HNSW')).toBeInTheDocument();
    });

    expect(screen.getByText('Index Management')).toBeInTheDocument();
    expect(screen.getByText('Ready')).toBeInTheDocument();
    expect(screen.getByText('M: 16, efConstruction: 200')).toBeInTheDocument();
  });

  test('allows creating a new index', async () => {
    render(<IndexManagement />);
    
    // Wait for existing indexes to load
    await waitFor(() => {
      expect(screen.getByText('HNSW')).toBeInTheDocument();
    });

    // Fill in the form to create a new index
    fireEvent.change(screen.getByLabelText('Index Type'), { target: { value: 'IVF' } });
    fireEvent.change(screen.getByLabelText('nlist'), { target: { value: '100' } });
    
    // Submit the form
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

    // Verify the new index is shown in the list
    expect(screen.getByText('IVF')).toBeInTheDocument();
  });

  test('allows deleting an index', async () => {
    render(<IndexManagement />);
    
    // Wait for indexes to load
    await waitFor(() => {
      expect(screen.getByText('HNSW')).toBeInTheDocument();
    });

    // Mock the delete API call
    indexApi.deleteIndex.mockResolvedValue({});

    // Find and click the delete button for the first index
    fireEvent.click(screen.getByRole('button', { name: /delete/i }));

    // Wait for confirmation dialog and confirm
    // (This would depend on how the confirmation is implemented in the actual component)
    // For now, let's just verify that deleteIndex would be called
    expect(screen.getByRole('button', { name: /delete/i })).toBeInTheDocument();
    
    // Simulate the confirmation
    indexApi.deleteIndex.mockResolvedValue({});
    fireEvent.click(screen.getByRole('button', { name: /delete/i }));
    
    await waitFor(() => {
      expect(indexApi.deleteIndex).toHaveBeenCalledWith('test-db-id', 'idx1');
    });
  });

  test('handles API errors gracefully', async () => {
    // Mock an API error for listing indexes
    indexApi.listIndexes.mockRejectedValue(new Error('Failed to fetch indexes'));
    
    render(<IndexManagement />);

    // Wait for the error handling to occur
    await waitFor(() => {
      expect(indexApi.listIndexes).toHaveBeenCalled();
    });
  });
});