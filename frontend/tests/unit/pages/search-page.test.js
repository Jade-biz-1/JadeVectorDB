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

// Mock next/router (not used by this page, but may be imported by Layout)
jest.mock('next/router', () => ({
  useRouter: () => ({
    query: {},
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
        {
          vector: { id: 'vec1', metadata: { tag: 'example' } },
          score: 0.95
        },
        {
          vector: { id: 'vec2', metadata: { tag: 'test' } },
          score: 0.85
        }
      ],
      queryTimeMs: 15.2,
      indexUsed: 'HNSW'
    });
  });

  test('loads and displays search form', async () => {
    render(<SearchPage />);

    // Wait for database list to load
    await waitFor(() => {
      expect(screen.getByText('Test DB')).toBeInTheDocument();
    });

    // Check form elements by their labels (matching actual implementation)
    expect(screen.getByLabelText(/Query Vector/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Top K Results/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Similarity Threshold/i)).toBeInTheDocument();
  });

  test('allows performing similarity search', async () => {
    render(<SearchPage />);

    // Wait for database list to load
    await waitFor(() => {
      expect(screen.getByText('Test DB')).toBeInTheDocument();
    });

    // Select database first (required)
    const dbSelect = screen.getByLabelText(/Database/i);
    fireEvent.change(dbSelect, { target: { value: 'test-db-id' } });

    // Fill in the query vector
    const queryInput = screen.getByLabelText(/Query Vector/i);
    fireEvent.change(queryInput, { target: { value: '0.1,0.2,0.3,0.4' } });

    // Set top K
    fireEvent.change(screen.getByLabelText(/Top K Results/i), { target: { value: '5' } });

    // Submit the search - button says "Search Similar Vectors"
    fireEvent.click(screen.getByRole('button', { name: /search similar vectors/i }));

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
    await waitFor(() => {
      expect(screen.getByText('Search Results')).toBeInTheDocument();
    });
  });

  test('handles search errors gracefully', async () => {
    // Mock a search error
    searchApi.similaritySearch.mockRejectedValue(new Error('Search failed'));

    render(<SearchPage />);

    // Wait for database list to load
    await waitFor(() => {
      expect(screen.getByText('Test DB')).toBeInTheDocument();
    });

    // Select database
    fireEvent.change(screen.getByLabelText(/Database/i), { target: { value: 'test-db-id' } });

    // Fill in the query vector
    fireEvent.change(screen.getByLabelText(/Query Vector/i), { target: { value: '0.1,0.2,0.3,0.4' } });

    // Submit the search
    fireEvent.click(screen.getByRole('button', { name: /search similar vectors/i }));

    // Wait for the error message to appear
    await waitFor(() => {
      expect(screen.getByText(/Error performing search/i)).toBeInTheDocument();
    });
  });

  // ============================================================================
  // Search Result Rendering Toggle Tests (T233)
  // ============================================================================

  describe('Search Result Rendering Toggles', () => {
    test('includeMetadata toggle is enabled by default', async () => {
      render(<SearchPage />);

      await waitFor(() => {
        expect(screen.getByText('Test DB')).toBeInTheDocument();
      });

      const metadataCheckbox = screen.getByLabelText(/include metadata/i);
      expect(metadataCheckbox).toBeChecked();
    });

    test('sends includeMetadata=true when checkbox is checked', async () => {
      render(<SearchPage />);

      await waitFor(() => {
        expect(screen.getByText('Test DB')).toBeInTheDocument();
      });

      // Select database
      fireEvent.change(screen.getByLabelText(/Database/i), { target: { value: 'test-db-id' } });

      // Fill in query vector
      fireEvent.change(screen.getByLabelText(/Query Vector/i), { target: { value: '0.1,0.2,0.3,0.4' } });

      // Ensure metadata checkbox is checked
      const metadataCheckbox = screen.getByLabelText(/include metadata/i);
      if (!metadataCheckbox.checked) {
        fireEvent.click(metadataCheckbox);
      }

      // Submit search
      fireEvent.click(screen.getByRole('button', { name: /search similar vectors/i }));

      await waitFor(() => {
        expect(searchApi.similaritySearch).toHaveBeenCalledWith(
          'test-db-id',
          expect.objectContaining({
            includeMetadata: true
          })
        );
      });
    });

    test('sends includeMetadata=false when checkbox is unchecked', async () => {
      render(<SearchPage />);

      await waitFor(() => {
        expect(screen.getByText('Test DB')).toBeInTheDocument();
      });

      // Select database
      fireEvent.change(screen.getByLabelText(/Database/i), { target: { value: 'test-db-id' } });

      // Fill in query vector
      fireEvent.change(screen.getByLabelText(/Query Vector/i), { target: { value: '0.1,0.2,0.3,0.4' } });

      // Uncheck metadata checkbox
      const metadataCheckbox = screen.getByLabelText(/include metadata/i);
      if (metadataCheckbox.checked) {
        fireEvent.click(metadataCheckbox);
      }

      // Submit search
      fireEvent.click(screen.getByRole('button', { name: /search similar vectors/i }));

      await waitFor(() => {
        expect(searchApi.similaritySearch).toHaveBeenCalledWith(
          'test-db-id',
          expect.objectContaining({
            includeMetadata: false
          })
        );
      });
    });

    test('sends includeVectorData=true when include values is checked', async () => {
      render(<SearchPage />);

      await waitFor(() => {
        expect(screen.getByText('Test DB')).toBeInTheDocument();
      });

      // Select database
      fireEvent.change(screen.getByLabelText(/Database/i), { target: { value: 'test-db-id' } });

      // Fill in query vector
      fireEvent.change(screen.getByLabelText(/Query Vector/i), { target: { value: '0.1,0.2,0.3,0.4' } });

      // Check include values checkbox
      const valuesCheckbox = screen.getByLabelText(/include vector values/i);
      if (!valuesCheckbox.checked) {
        fireEvent.click(valuesCheckbox);
      }

      // Submit search
      fireEvent.click(screen.getByRole('button', { name: /search similar vectors/i }));

      await waitFor(() => {
        expect(searchApi.similaritySearch).toHaveBeenCalledWith(
          'test-db-id',
          expect.objectContaining({
            includeVectorData: true
          })
        );
      });
    });

    test('displays results after search', async () => {
      render(<SearchPage />);

      await waitFor(() => {
        expect(screen.getByText('Test DB')).toBeInTheDocument();
      });

      // Select database
      fireEvent.change(screen.getByLabelText(/Database/i), { target: { value: 'test-db-id' } });

      // Fill in query vector and search
      fireEvent.change(screen.getByLabelText(/Query Vector/i), { target: { value: '0.1,0.2,0.3,0.4' } });
      fireEvent.click(screen.getByRole('button', { name: /search similar vectors/i }));

      await waitFor(() => {
        expect(screen.getByText('Search Results')).toBeInTheDocument();
      });
    });

    test('results display vector IDs', async () => {
      render(<SearchPage />);

      await waitFor(() => {
        expect(screen.getByText('Test DB')).toBeInTheDocument();
      });

      // Select database
      fireEvent.change(screen.getByLabelText(/Database/i), { target: { value: 'test-db-id' } });

      // Fill in query vector and search
      fireEvent.change(screen.getByLabelText(/Query Vector/i), { target: { value: '0.1,0.2,0.3,0.4' } });
      fireEvent.click(screen.getByRole('button', { name: /search similar vectors/i }));

      await waitFor(() => {
        // Vector IDs should be displayed (actual format is "Vector ID: vec1")
        expect(screen.getByText(/vec1/)).toBeInTheDocument();
      });
    });
  });
});
