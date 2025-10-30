// frontend/tests/unit/pages/search-page.test.js
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import SearchPage from '@/pages/search';

// Mock the API functions
jest.mock('@/lib/api', () => ({
  searchApi: {
    similaritySearch: jest.fn(),
    advancedSearch: jest.fn(),
  },
  databaseApi: {
    listDatabases: jest.fn(),
  }
}));

import { searchApi, databaseApi } from '@/lib/api';

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

describe('Search Page', () => {
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
    
    searchApi.similaritySearch.mockResolvedValue({
      results: [
        { id: 'vec1', similarity: 0.95, metadata: { tag: 'example' } },
        { id: 'vec2', similarity: 0.85, metadata: { tag: 'test' } }
      ],
      queryTimeMs: 15.2,
      indexUsed: 'HNSW'
    });
  });

  test('loads and displays search form', async () => {
    render(<SearchPage />);
    
    // Wait for database to load
    await waitFor(() => {
      expect(screen.getByText('Test DB')).toBeInTheDocument();
    });

    expect(screen.getByLabelText('Query Vector')).toBeInTheDocument();
    expect(screen.getByLabelText('Top K')).toBeInTheDocument();
    expect(screen.getByLabelText('Threshold')).toBeInTheDocument();
  });

  test('allows performing similarity search', async () => {
    render(<SearchPage />);
    
    // Wait for database to load
    await waitFor(() => {
      expect(screen.getByText('Test DB')).toBeInTheDocument();
    });

    // Fill in the query vector
    const queryInput = screen.getByLabelText('Query Vector');
    fireEvent.change(queryInput, { target: { value: '0.1,0.2,0.3,0.4' } });
    
    // Set top K
    fireEvent.change(screen.getByLabelText('Top K'), { target: { value: '5' } });
    
    // Submit the search
    fireEvent.click(screen.getByRole('button', { name: /perform search/i }));

    // Wait for the search to complete
    await waitFor(() => {
      expect(searchApi.similaritySearch).toHaveBeenCalledWith(
        'test-db-id', 
        expect.objectContaining({
          queryVector: [0.1, 0.2, 0.3, 0.4],
          topK: 5
        })
      );
    });

    // Verify results are displayed
    expect(screen.getByText('Search Results')).toBeInTheDocument();
    expect(screen.getByText('vec1')).toBeInTheDocument();
    expect(screen.getByText('vec2')).toBeInTheDocument();
  });

  test('handles search errors gracefully', async () => {
    // Mock a search error
    searchApi.similaritySearch.mockRejectedValue(new Error('Search failed'));
    
    render(<SearchPage />);
    
    // Wait for database to load
    await waitFor(() => {
      expect(screen.getByText('Test DB')).toBeInTheDocument();
    });

    // Fill in the query vector
    fireEvent.change(screen.getByLabelText('Query Vector'), { target: { value: '0.1,0.2,0.3,0.4' } });
    
    // Submit the search
    fireEvent.click(screen.getByRole('button', { name: /perform search/i }));

    // Wait for the error to be handled
    await waitFor(() => {
      expect(screen.queryByText('Search Results')).not.toBeInTheDocument();
    });
  });
});