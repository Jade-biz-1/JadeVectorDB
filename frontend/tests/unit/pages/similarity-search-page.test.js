// frontend/tests/unit/pages/similarity-search-page.test.js
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import SimilaritySearchPage from '@/pages/similarity-search';

// Mock the API functions
jest.mock('@/lib/api', () => ({
  searchApi: {
    similaritySearch: jest.fn(),
  },
  databaseApi: {
    listDatabases: jest.fn(),
  }
}));

import { searchApi, databaseApi } from '@/lib/api';

// Mock localStorage
Object.defineProperty(window, 'localStorage', {
  value: {
    getItem: jest.fn(() => 'test-auth-token'),
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

// Mock alert
beforeAll(() => {
  jest.spyOn(window, 'alert').mockImplementation(() => {});
});

// Mock performance.now
const mockPerformanceNow = jest.fn();
global.performance = { now: mockPerformanceNow };

describe('Similarity Search Page', () => {
  const mockDatabases = {
    databases: [
      { databaseId: 'db-1', name: 'Production DB', vectorDimension: 128 },
      { databaseId: 'db-2', name: 'Test DB', vectorDimension: 256 }
    ]
  };

  const mockSearchResults = {
    results: [
      { vectorId: 'vec-1', score: 0.95, values: [0.1, 0.2, 0.3], metadata: { title: 'Result 1' } },
      { vectorId: 'vec-2', score: 0.87, values: [0.4, 0.5, 0.6], metadata: { title: 'Result 2' } }
    ]
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockPerformanceNow.mockReturnValueOnce(0).mockReturnValueOnce(150); // 150ms search time

    // Mock successful API responses
    databaseApi.listDatabases.mockResolvedValue(mockDatabases);
    searchApi.similaritySearch.mockResolvedValue(mockSearchResults);
  });

  describe('Rendering', () => {
    test('renders database selector', async () => {
      render(<SimilaritySearchPage />);
      expect(screen.getByText('Select Database')).toBeInTheDocument();
      expect(screen.getByRole('combobox')).toBeInTheDocument();
    });

    test('renders query vector textarea', async () => {
      render(<SimilaritySearchPage />);
      expect(screen.getByText('Query Vector')).toBeInTheDocument();
      expect(screen.getByPlaceholderText(/enter comma-separated values/i)).toBeInTheDocument();
    });

    test('renders top k input', async () => {
      render(<SimilaritySearchPage />);
      expect(screen.getByText('Top K Results')).toBeInTheDocument();
      expect(screen.getByDisplayValue('10')).toBeInTheDocument();
    });

    test('renders threshold input', async () => {
      render(<SimilaritySearchPage />);
      expect(screen.getByText(/similarity threshold/i)).toBeInTheDocument();
      expect(screen.getByDisplayValue('0')).toBeInTheDocument();
    });

    test('renders search button', async () => {
      render(<SimilaritySearchPage />);
      expect(screen.getByRole('button', { name: /search/i })).toBeInTheDocument();
    });
  });

  describe('Database Selection', () => {
    test('fetches databases on mount', async () => {
      render(<SimilaritySearchPage />);

      await waitFor(() => {
        expect(databaseApi.listDatabases).toHaveBeenCalled();
      });
    });

    test('displays databases in dropdown', async () => {
      render(<SimilaritySearchPage />);

      await waitFor(() => {
        expect(screen.getByText(/Production DB/)).toBeInTheDocument();
        expect(screen.getByText(/Test DB/)).toBeInTheDocument();
      });
    });

    test('auto-selects first database', async () => {
      render(<SimilaritySearchPage />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveValue('db-1');
      });
    });

    test('shows alert on database fetch error', async () => {
      databaseApi.listDatabases.mockRejectedValue(new Error('Network error'));

      render(<SimilaritySearchPage />);

      await waitFor(() => {
        expect(window.alert).toHaveBeenCalledWith(expect.stringContaining('Network error'));
      });
    });
  });

  describe('Vector Input Parsing', () => {
    test('parses comma-separated values', async () => {
      render(<SimilaritySearchPage />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveValue('db-1');
      });

      const vectorInput = screen.getByPlaceholderText(/enter comma-separated values/i);
      fireEvent.change(vectorInput, { target: { value: '0.1, 0.2, 0.3' } });

      fireEvent.click(screen.getByRole('button', { name: /search/i }));

      await waitFor(() => {
        expect(searchApi.similaritySearch).toHaveBeenCalledWith('db-1', expect.objectContaining({
          queryVector: [0.1, 0.2, 0.3]
        }));
      });
    });

    test('parses JSON array format', async () => {
      render(<SimilaritySearchPage />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveValue('db-1');
      });

      const vectorInput = screen.getByPlaceholderText(/enter comma-separated values/i);
      fireEvent.change(vectorInput, { target: { value: '[0.1, 0.2, 0.3, 0.4]' } });

      fireEvent.click(screen.getByRole('button', { name: /search/i }));

      await waitFor(() => {
        expect(searchApi.similaritySearch).toHaveBeenCalledWith('db-1', expect.objectContaining({
          queryVector: [0.1, 0.2, 0.3, 0.4]
        }));
      });
    });

    test('shows alert for empty vector', async () => {
      render(<SimilaritySearchPage />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveValue('db-1');
      });

      const vectorInput = screen.getByPlaceholderText(/enter comma-separated values/i);
      fireEvent.change(vectorInput, { target: { value: 'not, valid, numbers' } });

      fireEvent.click(screen.getByRole('button', { name: /search/i }));

      await waitFor(() => {
        expect(window.alert).toHaveBeenCalledWith('Please enter a valid vector');
      });
    });
  });

  describe('Search Execution', () => {
    test('calls search API with parameters', async () => {
      render(<SimilaritySearchPage />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveValue('db-1');
      });

      const vectorInput = screen.getByPlaceholderText(/enter comma-separated values/i);
      fireEvent.change(vectorInput, { target: { value: '0.1, 0.2, 0.3' } });

      fireEvent.click(screen.getByRole('button', { name: /search/i }));

      await waitFor(() => {
        expect(searchApi.similaritySearch).toHaveBeenCalledWith('db-1', {
          queryVector: [0.1, 0.2, 0.3],
          topK: 10,
          threshold: 0
        });
      });
    });

    test('uses custom top-k and threshold values', async () => {
      render(<SimilaritySearchPage />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveValue('db-1');
      });

      const vectorInput = screen.getByPlaceholderText(/enter comma-separated values/i);
      fireEvent.change(vectorInput, { target: { value: '0.1, 0.2' } });

      const topKInput = screen.getByDisplayValue('10');
      fireEvent.change(topKInput, { target: { value: '20' } });

      const thresholdInput = screen.getByDisplayValue('0');
      fireEvent.change(thresholdInput, { target: { value: '0.5' } });

      fireEvent.click(screen.getByRole('button', { name: /search/i }));

      await waitFor(() => {
        expect(searchApi.similaritySearch).toHaveBeenCalledWith('db-1', expect.objectContaining({
          topK: 20,
          threshold: 0.5
        }));
      });
    });

    test('shows loading state during search', async () => {
      searchApi.similaritySearch.mockImplementation(() => new Promise(() => {}));

      render(<SimilaritySearchPage />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveValue('db-1');
      });

      const vectorInput = screen.getByPlaceholderText(/enter comma-separated values/i);
      fireEvent.change(vectorInput, { target: { value: '0.1, 0.2' } });

      fireEvent.click(screen.getByRole('button', { name: /search/i }));

      expect(screen.getByRole('button', { name: /searching/i })).toBeInTheDocument();
    });

    test('disables button during loading', async () => {
      searchApi.similaritySearch.mockImplementation(() => new Promise(() => {}));

      render(<SimilaritySearchPage />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveValue('db-1');
      });

      const vectorInput = screen.getByPlaceholderText(/enter comma-separated values/i);
      fireEvent.change(vectorInput, { target: { value: '0.1, 0.2' } });

      fireEvent.click(screen.getByRole('button', { name: /search/i }));

      expect(screen.getByRole('button', { name: /searching/i })).toBeDisabled();
    });

    // Button is disabled when no database selected, so validation through alert doesn't trigger
    // Test is covered by "disables search button when no database selected"
  });

  describe('Results Display', () => {
    test('displays results count', async () => {
      render(<SimilaritySearchPage />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveValue('db-1');
      });

      const vectorInput = screen.getByPlaceholderText(/enter comma-separated values/i);
      fireEvent.change(vectorInput, { target: { value: '0.1, 0.2' } });

      fireEvent.click(screen.getByRole('button', { name: /search/i }));

      await waitFor(() => {
        expect(screen.getByText(/Search Results \(2\)/)).toBeInTheDocument();
      });
    });

    test('displays search time', async () => {
      render(<SimilaritySearchPage />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveValue('db-1');
      });

      const vectorInput = screen.getByPlaceholderText(/enter comma-separated values/i);
      fireEvent.change(vectorInput, { target: { value: '0.1, 0.2' } });

      fireEvent.click(screen.getByRole('button', { name: /search/i }));

      await waitFor(() => {
        expect(screen.getByText(/Search completed in/)).toBeInTheDocument();
      });
    });

    test('displays result rankings', async () => {
      render(<SimilaritySearchPage />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveValue('db-1');
      });

      const vectorInput = screen.getByPlaceholderText(/enter comma-separated values/i);
      fireEvent.change(vectorInput, { target: { value: '0.1, 0.2' } });

      fireEvent.click(screen.getByRole('button', { name: /search/i }));

      await waitFor(() => {
        expect(screen.getByText('Rank #1')).toBeInTheDocument();
        expect(screen.getByText('Rank #2')).toBeInTheDocument();
      });
    });

    test('displays similarity scores', async () => {
      render(<SimilaritySearchPage />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveValue('db-1');
      });

      const vectorInput = screen.getByPlaceholderText(/enter comma-separated values/i);
      fireEvent.change(vectorInput, { target: { value: '0.1, 0.2' } });

      fireEvent.click(screen.getByRole('button', { name: /search/i }));

      await waitFor(() => {
        expect(screen.getByText('0.9500')).toBeInTheDocument();
        expect(screen.getByText('0.8700')).toBeInTheDocument();
      });
    });

    test('displays vector IDs', async () => {
      render(<SimilaritySearchPage />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveValue('db-1');
      });

      const vectorInput = screen.getByPlaceholderText(/enter comma-separated values/i);
      fireEvent.change(vectorInput, { target: { value: '0.1, 0.2' } });

      fireEvent.click(screen.getByRole('button', { name: /search/i }));

      await waitFor(() => {
        expect(screen.getByText(/ID: vec-1/)).toBeInTheDocument();
        expect(screen.getByText(/ID: vec-2/)).toBeInTheDocument();
      });
    });

    test('displays no results message', async () => {
      searchApi.similaritySearch.mockResolvedValue({ results: [] });

      render(<SimilaritySearchPage />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveValue('db-1');
      });

      const vectorInput = screen.getByPlaceholderText(/enter comma-separated values/i);
      fireEvent.change(vectorInput, { target: { value: '0.1, 0.2' } });

      fireEvent.click(screen.getByRole('button', { name: /search/i }));

      await waitFor(() => {
        expect(screen.getByText(/No results found matching your criteria/)).toBeInTheDocument();
      });
    });
  });

  describe('Error Handling', () => {
    test('shows alert on search error', async () => {
      searchApi.similaritySearch.mockRejectedValue(new Error('Search failed'));

      render(<SimilaritySearchPage />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveValue('db-1');
      });

      const vectorInput = screen.getByPlaceholderText(/enter comma-separated values/i);
      fireEvent.change(vectorInput, { target: { value: '0.1, 0.2' } });

      fireEvent.click(screen.getByRole('button', { name: /search/i }));

      await waitFor(() => {
        expect(window.alert).toHaveBeenCalledWith(expect.stringContaining('Search failed'));
      });
    });

    test('clears results on error', async () => {
      // First successful search
      render(<SimilaritySearchPage />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveValue('db-1');
      });

      const vectorInput = screen.getByPlaceholderText(/enter comma-separated values/i);
      fireEvent.change(vectorInput, { target: { value: '0.1, 0.2' } });

      fireEvent.click(screen.getByRole('button', { name: /search/i }));

      await waitFor(() => {
        expect(screen.getByText(/Search Results \(2\)/)).toBeInTheDocument();
      });

      // Now trigger an error
      searchApi.similaritySearch.mockRejectedValue(new Error('Search failed'));
      mockPerformanceNow.mockReturnValueOnce(0).mockReturnValueOnce(100);

      fireEvent.click(screen.getByRole('button', { name: /search/i }));

      await waitFor(() => {
        expect(screen.getByText(/Search Results \(0\)/)).toBeInTheDocument();
      });
    });
  });

  describe('Button State', () => {
    test('disables search button when no database selected', async () => {
      databaseApi.listDatabases.mockResolvedValue({ databases: [] });

      render(<SimilaritySearchPage />);

      await waitFor(() => {
        expect(databaseApi.listDatabases).toHaveBeenCalled();
      });

      expect(screen.getByRole('button', { name: /search/i })).toBeDisabled();
    });

    test('enables search button when database selected', async () => {
      render(<SimilaritySearchPage />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveValue('db-1');
      });

      expect(screen.getByRole('button', { name: /search/i })).not.toBeDisabled();
    });
  });
});
