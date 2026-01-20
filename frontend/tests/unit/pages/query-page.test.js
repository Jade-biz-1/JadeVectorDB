// frontend/tests/unit/pages/query-page.test.js
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import QueryInterface from '@/pages/query';

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

describe('Query Interface Page', () => {
  const mockDatabases = {
    databases: [
      { databaseId: 'db-1', name: 'Production DB', vectorDimension: 128 },
      { databaseId: 'db-2', name: 'Test DB', vectorDimension: 256 }
    ]
  };

  const mockSearchResults = {
    results: [
      { id: 'vec-1', score: 0.95, metadata: { title: 'Result 1' } },
      { id: 'vec-2', score: 0.87, metadata: { title: 'Result 2' } },
      { id: 'vec-3', score: 0.82, metadata: { title: 'Result 3' } }
    ],
    query: 'test query',
    totalResults: 3
  };

  beforeEach(() => {
    jest.clearAllMocks();

    // Mock successful API responses
    databaseApi.listDatabases.mockResolvedValue(mockDatabases);
    searchApi.similaritySearch.mockResolvedValue(mockSearchResults);
  });

  describe('Rendering', () => {
    test('renders page title', async () => {
      render(<QueryInterface />);
      // "Run Query" appears in both h2 heading and button
      expect(screen.getByRole('heading', { name: /run query/i })).toBeInTheDocument();
    });

    test('renders database selector', async () => {
      render(<QueryInterface />);
      expect(screen.getByText('Select Database')).toBeInTheDocument();
      expect(screen.getByRole('combobox')).toBeInTheDocument();
    });

    test('renders query input', async () => {
      render(<QueryInterface />);
      expect(screen.getByText('Query')).toBeInTheDocument();
      expect(screen.getByPlaceholderText(/enter your search query/i)).toBeInTheDocument();
    });

    test('renders top-k input', async () => {
      render(<QueryInterface />);
      expect(screen.getByText('Top-K')).toBeInTheDocument();
      expect(screen.getByDisplayValue('10')).toBeInTheDocument();
    });

    test('renders run query button', async () => {
      render(<QueryInterface />);
      expect(screen.getByRole('button', { name: /run query/i })).toBeInTheDocument();
    });
  });

  describe('Database Selection', () => {
    test('fetches databases on mount', async () => {
      render(<QueryInterface />);

      await waitFor(() => {
        expect(databaseApi.listDatabases).toHaveBeenCalled();
      });
    });

    test('displays databases in dropdown', async () => {
      render(<QueryInterface />);

      await waitFor(() => {
        expect(screen.getByText(/Production DB/)).toBeInTheDocument();
        expect(screen.getByText(/Test DB/)).toBeInTheDocument();
      });
    });

    test('shows dimension in database options', async () => {
      render(<QueryInterface />);

      await waitFor(() => {
        expect(screen.getByText(/Dimension: 128/)).toBeInTheDocument();
        expect(screen.getByText(/Dimension: 256/)).toBeInTheDocument();
      });
    });

    test('auto-selects first database', async () => {
      render(<QueryInterface />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveValue('db-1');
      });
    });

    test('shows alert on database fetch error', async () => {
      databaseApi.listDatabases.mockRejectedValue(new Error('Network error'));

      render(<QueryInterface />);

      await waitFor(() => {
        expect(window.alert).toHaveBeenCalledWith(expect.stringContaining('Network error'));
      });
    });
  });

  describe('Query Execution', () => {
    test('calls search API with query parameters', async () => {
      render(<QueryInterface />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveValue('db-1');
      });

      const queryInput = screen.getByPlaceholderText(/enter your search query/i);
      fireEvent.change(queryInput, { target: { value: 'test search query' } });

      fireEvent.click(screen.getByRole('button', { name: /run query/i }));

      await waitFor(() => {
        expect(searchApi.similaritySearch).toHaveBeenCalledWith('db-1', {
          query: 'test search query',
          topK: 10
        });
      });
    });

    test('uses custom top-k value', async () => {
      render(<QueryInterface />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveValue('db-1');
      });

      const queryInput = screen.getByPlaceholderText(/enter your search query/i);
      fireEvent.change(queryInput, { target: { value: 'test query' } });

      const topKInput = screen.getByDisplayValue('10');
      fireEvent.change(topKInput, { target: { value: '25' } });

      fireEvent.click(screen.getByRole('button', { name: /run query/i }));

      await waitFor(() => {
        expect(searchApi.similaritySearch).toHaveBeenCalledWith('db-1', {
          query: 'test query',
          topK: 25
        });
      });
    });

    test('shows loading state during query', async () => {
      searchApi.similaritySearch.mockImplementation(() => new Promise(() => {}));

      render(<QueryInterface />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveValue('db-1');
      });

      const queryInput = screen.getByPlaceholderText(/enter your search query/i);
      fireEvent.change(queryInput, { target: { value: 'test query' } });

      fireEvent.click(screen.getByRole('button', { name: /run query/i }));

      expect(screen.getByRole('button', { name: /running query/i })).toBeInTheDocument();
    });

    test('disables button during loading', async () => {
      searchApi.similaritySearch.mockImplementation(() => new Promise(() => {}));

      render(<QueryInterface />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveValue('db-1');
      });

      const queryInput = screen.getByPlaceholderText(/enter your search query/i);
      fireEvent.change(queryInput, { target: { value: 'test query' } });

      fireEvent.click(screen.getByRole('button', { name: /run query/i }));

      expect(screen.getByRole('button', { name: /running query/i })).toBeDisabled();
    });

    test('executes query on Enter key press', async () => {
      render(<QueryInterface />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveValue('db-1');
      });

      const queryInput = screen.getByPlaceholderText(/enter your search query/i);
      fireEvent.change(queryInput, { target: { value: 'test query' } });
      fireEvent.keyPress(queryInput, { key: 'Enter', code: 'Enter', charCode: 13 });

      await waitFor(() => {
        expect(searchApi.similaritySearch).toHaveBeenCalled();
      });
    });
  });

  describe('Form Validation', () => {
    // Button is disabled when no database selected, so validation through alert doesn't trigger
    // The validation is handled by the disabled state
    test('disables button when no database selected', async () => {
      databaseApi.listDatabases.mockResolvedValue({ databases: [] });

      render(<QueryInterface />);

      await waitFor(() => {
        expect(databaseApi.listDatabases).toHaveBeenCalled();
      });

      expect(screen.getByRole('button', { name: /run query/i })).toBeDisabled();
    });

    test('shows alert when query is empty', async () => {
      render(<QueryInterface />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveValue('db-1');
      });

      // Don't enter any query
      fireEvent.click(screen.getByRole('button', { name: /run query/i }));

      expect(window.alert).toHaveBeenCalledWith('Please enter a query');
      expect(searchApi.similaritySearch).not.toHaveBeenCalled();
    });

    test('shows alert when query is whitespace only', async () => {
      render(<QueryInterface />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveValue('db-1');
      });

      const queryInput = screen.getByPlaceholderText(/enter your search query/i);
      fireEvent.change(queryInput, { target: { value: '   ' } });

      fireEvent.click(screen.getByRole('button', { name: /run query/i }));

      expect(window.alert).toHaveBeenCalledWith('Please enter a query');
    });

  });

  describe('Results Display', () => {
    test('displays results after successful query', async () => {
      render(<QueryInterface />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveValue('db-1');
      });

      const queryInput = screen.getByPlaceholderText(/enter your search query/i);
      fireEvent.change(queryInput, { target: { value: 'test query' } });

      fireEvent.click(screen.getByRole('button', { name: /run query/i }));

      await waitFor(() => {
        expect(screen.getByText('Results')).toBeInTheDocument();
        expect(screen.getByText(/Found 3 results/)).toBeInTheDocument();
      });
    });

    test('displays result count', async () => {
      render(<QueryInterface />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveValue('db-1');
      });

      const queryInput = screen.getByPlaceholderText(/enter your search query/i);
      fireEvent.change(queryInput, { target: { value: 'test query' } });

      fireEvent.click(screen.getByRole('button', { name: /run query/i }));

      await waitFor(() => {
        expect(screen.getByText(/Found 3 results/)).toBeInTheDocument();
      });
    });

    test('displays JSON results', async () => {
      render(<QueryInterface />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveValue('db-1');
      });

      const queryInput = screen.getByPlaceholderText(/enter your search query/i);
      fireEvent.change(queryInput, { target: { value: 'test query' } });

      fireEvent.click(screen.getByRole('button', { name: /run query/i }));

      await waitFor(() => {
        // Check that result data is displayed
        expect(screen.getByText(/vec-1/)).toBeInTheDocument();
        expect(screen.getByText(/0.95/)).toBeInTheDocument();
      });
    });

    test('displays error message on query failure', async () => {
      searchApi.similaritySearch.mockRejectedValue(new Error('Search failed'));

      render(<QueryInterface />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveValue('db-1');
      });

      const queryInput = screen.getByPlaceholderText(/enter your search query/i);
      fireEvent.change(queryInput, { target: { value: 'test query' } });

      fireEvent.click(screen.getByRole('button', { name: /run query/i }));

      await waitFor(() => {
        expect(screen.getByText('Error:')).toBeInTheDocument();
        expect(screen.getByText('Search failed')).toBeInTheDocument();
      });
    });

    test('handles empty results', async () => {
      searchApi.similaritySearch.mockResolvedValue({ results: [] });

      render(<QueryInterface />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveValue('db-1');
      });

      const queryInput = screen.getByPlaceholderText(/enter your search query/i);
      fireEvent.change(queryInput, { target: { value: 'test query' } });

      fireEvent.click(screen.getByRole('button', { name: /run query/i }));

      await waitFor(() => {
        expect(screen.getByText(/Found 0 results/)).toBeInTheDocument();
      });
    });
  });

  describe('Database Change', () => {
    test('allows changing database selection', async () => {
      render(<QueryInterface />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveValue('db-1');
      });

      fireEvent.change(screen.getByRole('combobox'), { target: { value: 'db-2' } });

      expect(screen.getByRole('combobox')).toHaveValue('db-2');
    });

    test('uses selected database for query', async () => {
      render(<QueryInterface />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveValue('db-1');
      });

      // Change to second database
      fireEvent.change(screen.getByRole('combobox'), { target: { value: 'db-2' } });

      const queryInput = screen.getByPlaceholderText(/enter your search query/i);
      fireEvent.change(queryInput, { target: { value: 'test query' } });

      fireEvent.click(screen.getByRole('button', { name: /run query/i }));

      await waitFor(() => {
        expect(searchApi.similaritySearch).toHaveBeenCalledWith('db-2', expect.any(Object));
      });
    });
  });
});
