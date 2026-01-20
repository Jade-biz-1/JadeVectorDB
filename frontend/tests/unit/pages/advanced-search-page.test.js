// frontend/tests/unit/pages/advanced-search-page.test.js
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import AdvancedSearch from '@/pages/advanced-search';

// Mock the API functions
jest.mock('@/lib/api', () => ({
  searchApi: {
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

// Mock window.location
delete window.location;
window.location = { search: '' };

// Mock alert
beforeAll(() => {
  jest.spyOn(window, 'alert').mockImplementation(() => {});
});

describe('Advanced Search Page', () => {
  const mockDatabases = {
    databases: [
      { databaseId: 'db-1', name: 'Production DB' },
      { databaseId: 'db-2', name: 'Test DB' }
    ]
  };

  const mockSearchResults = {
    results: [
      {
        vectorId: 'vec-1',
        score: 0.95,
        vector: {
          id: 'vec-1',
          values: [0.1, 0.2, 0.3, 0.4, 0.5],
          metadata: { title: 'Result 1', category: 'A' }
        }
      },
      {
        vectorId: 'vec-2',
        score: 0.87,
        vector: {
          id: 'vec-2',
          values: [0.4, 0.5, 0.6, 0.7, 0.8],
          metadata: { title: 'Result 2', category: 'B' }
        }
      }
    ]
  };

  beforeEach(() => {
    jest.clearAllMocks();
    window.location.search = '';

    // Mock successful API responses
    databaseApi.listDatabases.mockResolvedValue(mockDatabases);
    searchApi.advancedSearch.mockResolvedValue(mockSearchResults);
  });

  describe('Rendering', () => {
    test('renders page title', () => {
      render(<AdvancedSearch />);
      expect(screen.getByText('Advanced Similarity Search')).toBeInTheDocument();
    });

    test('renders page description', () => {
      render(<AdvancedSearch />);
      expect(screen.getByText(/Find vectors similar to your query vector with metadata filters/)).toBeInTheDocument();
    });

    test('renders database selector', () => {
      render(<AdvancedSearch />);
      expect(screen.getByLabelText('Database')).toBeInTheDocument();
    });

    test('renders query vector textarea', () => {
      render(<AdvancedSearch />);
      expect(screen.getByLabelText('Query Vector')).toBeInTheDocument();
      expect(screen.getByPlaceholderText(/enter vector values as comma-separated numbers/i)).toBeInTheDocument();
    });

    test('renders top k input', () => {
      render(<AdvancedSearch />);
      expect(screen.getByLabelText('Top K Results')).toBeInTheDocument();
      expect(screen.getByDisplayValue('10')).toBeInTheDocument();
    });

    test('renders threshold input', () => {
      render(<AdvancedSearch />);
      expect(screen.getByLabelText('Similarity Threshold')).toBeInTheDocument();
    });

    test('renders include metadata checkbox', () => {
      render(<AdvancedSearch />);
      expect(screen.getByLabelText('Include metadata in results')).toBeInTheDocument();
    });

    test('renders include values checkbox', () => {
      render(<AdvancedSearch />);
      expect(screen.getByLabelText('Include vector values in results')).toBeInTheDocument();
    });

    test('renders metadata filters section', () => {
      render(<AdvancedSearch />);
      expect(screen.getByText('Metadata Filters')).toBeInTheDocument();
    });

    test('renders add filter button', () => {
      render(<AdvancedSearch />);
      expect(screen.getByRole('button', { name: /add filter/i })).toBeInTheDocument();
    });

    test('renders search button', () => {
      render(<AdvancedSearch />);
      expect(screen.getByRole('button', { name: /perform advanced search/i })).toBeInTheDocument();
    });

    test('defaults include metadata to checked', () => {
      render(<AdvancedSearch />);
      expect(screen.getByLabelText('Include metadata in results')).toBeChecked();
    });

    test('defaults include values to unchecked', () => {
      render(<AdvancedSearch />);
      expect(screen.getByLabelText('Include vector values in results')).not.toBeChecked();
    });
  });

  describe('Database Selection', () => {
    test('fetches databases on mount', async () => {
      render(<AdvancedSearch />);

      await waitFor(() => {
        expect(databaseApi.listDatabases).toHaveBeenCalled();
      });
    });

    test('displays databases in dropdown', async () => {
      render(<AdvancedSearch />);

      await waitFor(() => {
        expect(screen.getByText('Production DB')).toBeInTheDocument();
        expect(screen.getByText('Test DB')).toBeInTheDocument();
      });
    });

    test('shows alert on database fetch error', async () => {
      databaseApi.listDatabases.mockRejectedValue(new Error('Network error'));

      render(<AdvancedSearch />);

      await waitFor(() => {
        expect(window.alert).toHaveBeenCalledWith(expect.stringContaining('Network error'));
      });
    });

    test('reads database ID from URL parameters', async () => {
      window.location.search = '?databaseId=db-2';

      render(<AdvancedSearch />);

      await waitFor(() => {
        expect(screen.getByLabelText('Database')).toHaveValue('db-2');
      });
    });
  });

  describe('Filter Management', () => {
    test('renders one filter row by default', () => {
      render(<AdvancedSearch />);
      expect(screen.getAllByPlaceholderText('Field name')).toHaveLength(1);
    });

    test('adds new filter on button click', () => {
      render(<AdvancedSearch />);

      fireEvent.click(screen.getByRole('button', { name: /add filter/i }));

      expect(screen.getAllByPlaceholderText('Field name')).toHaveLength(2);
    });

    test('removes filter on remove button click', () => {
      render(<AdvancedSearch />);

      // Add a second filter first
      fireEvent.click(screen.getByRole('button', { name: /add filter/i }));
      expect(screen.getAllByPlaceholderText('Field name')).toHaveLength(2);

      // Remove one filter
      const removeButtons = screen.getAllByRole('button').filter(btn =>
        btn.querySelector('svg') && btn.closest('.col-span-1')
      );
      fireEvent.click(removeButtons[0]);

      expect(screen.getAllByPlaceholderText('Field name')).toHaveLength(1);
    });

    test('does not remove last filter', () => {
      render(<AdvancedSearch />);

      // Try to remove the only filter
      const removeButtons = screen.getAllByRole('button').filter(btn =>
        btn.querySelector('svg') && btn.closest('.col-span-1')
      );

      expect(removeButtons[0]).toBeDisabled();
    });

    test('updates filter field value', () => {
      render(<AdvancedSearch />);

      const fieldInput = screen.getByPlaceholderText('Field name');
      fireEvent.change(fieldInput, { target: { value: 'category' } });

      expect(fieldInput).toHaveValue('category');
    });

    test('updates filter operator', () => {
      render(<AdvancedSearch />);

      const operatorSelect = screen.getByDisplayValue('Equals');
      fireEvent.change(operatorSelect, { target: { value: 'contains' } });

      expect(operatorSelect).toHaveValue('contains');
    });

    test('updates filter value', () => {
      render(<AdvancedSearch />);

      const valueInput = screen.getByPlaceholderText('Value');
      fireEvent.change(valueInput, { target: { value: 'test-value' } });

      expect(valueInput).toHaveValue('test-value');
    });

    test('renders all operator options', () => {
      render(<AdvancedSearch />);

      expect(screen.getByText('Equals')).toBeInTheDocument();
      expect(screen.getByText('Contains')).toBeInTheDocument();
      expect(screen.getByText('Greater than')).toBeInTheDocument();
      expect(screen.getByText('Less than')).toBeInTheDocument();
    });
  });

  describe('Include Options', () => {
    test('toggles include metadata', () => {
      render(<AdvancedSearch />);

      const checkbox = screen.getByLabelText('Include metadata in results');
      expect(checkbox).toBeChecked();

      fireEvent.click(checkbox);
      expect(checkbox).not.toBeChecked();
    });

    test('toggles include values', () => {
      render(<AdvancedSearch />);

      const checkbox = screen.getByLabelText('Include vector values in results');
      expect(checkbox).not.toBeChecked();

      fireEvent.click(checkbox);
      expect(checkbox).toBeChecked();
    });
  });

  describe('Search Execution', () => {
    test('calls search API with parameters', async () => {
      render(<AdvancedSearch />);

      await waitFor(() => {
        expect(screen.getByText('Production DB')).toBeInTheDocument();
      });

      // Select database
      fireEvent.change(screen.getByLabelText('Database'), { target: { value: 'db-1' } });

      // Enter query vector
      const vectorInput = screen.getByLabelText('Query Vector');
      fireEvent.change(vectorInput, { target: { value: '0.1, 0.2, 0.3' } });

      // Submit search
      fireEvent.click(screen.getByRole('button', { name: /perform advanced search/i }));

      await waitFor(() => {
        expect(searchApi.advancedSearch).toHaveBeenCalledWith('db-1', expect.objectContaining({
          queryVector: [0.1, 0.2, 0.3],
          topK: 10,
          threshold: 0,
          includeMetadata: true,
          includeVectorData: false,
          includeValues: false
        }));
      });
    });

    test('parses JSON array vector format', async () => {
      render(<AdvancedSearch />);

      await waitFor(() => {
        expect(screen.getByText('Production DB')).toBeInTheDocument();
      });

      fireEvent.change(screen.getByLabelText('Database'), { target: { value: 'db-1' } });

      const vectorInput = screen.getByLabelText('Query Vector');
      fireEvent.change(vectorInput, { target: { value: '[0.1, 0.2, 0.3, 0.4]' } });

      fireEvent.click(screen.getByRole('button', { name: /perform advanced search/i }));

      await waitFor(() => {
        expect(searchApi.advancedSearch).toHaveBeenCalledWith('db-1', expect.objectContaining({
          queryVector: [0.1, 0.2, 0.3, 0.4]
        }));
      });
    });

    test('includes filters in search request', async () => {
      render(<AdvancedSearch />);

      await waitFor(() => {
        expect(screen.getByText('Production DB')).toBeInTheDocument();
      });

      fireEvent.change(screen.getByLabelText('Database'), { target: { value: 'db-1' } });

      const vectorInput = screen.getByLabelText('Query Vector');
      fireEvent.change(vectorInput, { target: { value: '0.1, 0.2' } });

      // Add filter
      fireEvent.change(screen.getByPlaceholderText('Field name'), { target: { value: 'category' } });
      fireEvent.change(screen.getByPlaceholderText('Value'), { target: { value: 'test' } });

      fireEvent.click(screen.getByRole('button', { name: /perform advanced search/i }));

      await waitFor(() => {
        expect(searchApi.advancedSearch).toHaveBeenCalledWith('db-1', expect.objectContaining({
          filters: { category: 'test' }
        }));
      });
    });

    test('shows loading state during search', async () => {
      searchApi.advancedSearch.mockImplementation(() => new Promise(() => {}));

      render(<AdvancedSearch />);

      await waitFor(() => {
        expect(screen.getByText('Production DB')).toBeInTheDocument();
      });

      fireEvent.change(screen.getByLabelText('Database'), { target: { value: 'db-1' } });

      const vectorInput = screen.getByLabelText('Query Vector');
      fireEvent.change(vectorInput, { target: { value: '0.1, 0.2' } });

      fireEvent.click(screen.getByRole('button', { name: /perform advanced search/i }));

      expect(screen.getByRole('button', { name: /searching/i })).toBeInTheDocument();
    });

    test('disables button during loading', async () => {
      searchApi.advancedSearch.mockImplementation(() => new Promise(() => {}));

      render(<AdvancedSearch />);

      await waitFor(() => {
        expect(screen.getByText('Production DB')).toBeInTheDocument();
      });

      fireEvent.change(screen.getByLabelText('Database'), { target: { value: 'db-1' } });

      const vectorInput = screen.getByLabelText('Query Vector');
      fireEvent.change(vectorInput, { target: { value: '0.1, 0.2' } });

      fireEvent.click(screen.getByRole('button', { name: /perform advanced search/i }));

      expect(screen.getByRole('button', { name: /searching/i })).toBeDisabled();
    });

    test('disables button when no database selected', async () => {
      render(<AdvancedSearch />);

      await waitFor(() => {
        expect(databaseApi.listDatabases).toHaveBeenCalled();
      });

      const vectorInput = screen.getByLabelText('Query Vector');
      fireEvent.change(vectorInput, { target: { value: '0.1, 0.2' } });

      expect(screen.getByRole('button', { name: /perform advanced search/i })).toBeDisabled();
    });

    test('disables button when no query vector', async () => {
      render(<AdvancedSearch />);

      await waitFor(() => {
        expect(screen.getByText('Production DB')).toBeInTheDocument();
      });

      fireEvent.change(screen.getByLabelText('Database'), { target: { value: 'db-1' } });

      expect(screen.getByRole('button', { name: /perform advanced search/i })).toBeDisabled();
    });
  });

  describe('Results Display', () => {
    test('displays search results', async () => {
      render(<AdvancedSearch />);

      await waitFor(() => {
        expect(screen.getByText('Production DB')).toBeInTheDocument();
      });

      fireEvent.change(screen.getByLabelText('Database'), { target: { value: 'db-1' } });

      const vectorInput = screen.getByLabelText('Query Vector');
      fireEvent.change(vectorInput, { target: { value: '0.1, 0.2' } });

      fireEvent.click(screen.getByRole('button', { name: /perform advanced search/i }));

      await waitFor(() => {
        expect(screen.getByText('Search Results')).toBeInTheDocument();
      });
    });

    test('displays result count', async () => {
      render(<AdvancedSearch />);

      await waitFor(() => {
        expect(screen.getByText('Production DB')).toBeInTheDocument();
      });

      fireEvent.change(screen.getByLabelText('Database'), { target: { value: 'db-1' } });

      const vectorInput = screen.getByLabelText('Query Vector');
      fireEvent.change(vectorInput, { target: { value: '0.1, 0.2' } });

      fireEvent.click(screen.getByRole('button', { name: /perform advanced search/i }));

      await waitFor(() => {
        expect(screen.getByText(/Top 2 most similar vectors/)).toBeInTheDocument();
      });
    });

    test('displays vector IDs', async () => {
      render(<AdvancedSearch />);

      await waitFor(() => {
        expect(screen.getByText('Production DB')).toBeInTheDocument();
      });

      fireEvent.change(screen.getByLabelText('Database'), { target: { value: 'db-1' } });

      const vectorInput = screen.getByLabelText('Query Vector');
      fireEvent.change(vectorInput, { target: { value: '0.1, 0.2' } });

      fireEvent.click(screen.getByRole('button', { name: /perform advanced search/i }));

      await waitFor(() => {
        expect(screen.getByText(/Vector ID: vec-1/)).toBeInTheDocument();
        expect(screen.getByText(/Vector ID: vec-2/)).toBeInTheDocument();
      });
    });

    test('displays similarity scores', async () => {
      render(<AdvancedSearch />);

      await waitFor(() => {
        expect(screen.getByText('Production DB')).toBeInTheDocument();
      });

      fireEvent.change(screen.getByLabelText('Database'), { target: { value: 'db-1' } });

      const vectorInput = screen.getByLabelText('Query Vector');
      fireEvent.change(vectorInput, { target: { value: '0.1, 0.2' } });

      fireEvent.click(screen.getByRole('button', { name: /perform advanced search/i }));

      await waitFor(() => {
        expect(screen.getByText(/0.9500 similarity/)).toBeInTheDocument();
        expect(screen.getByText(/0.8700 similarity/)).toBeInTheDocument();
      });
    });

    test('displays metadata when present', async () => {
      render(<AdvancedSearch />);

      await waitFor(() => {
        expect(screen.getByText('Production DB')).toBeInTheDocument();
      });

      fireEvent.change(screen.getByLabelText('Database'), { target: { value: 'db-1' } });

      const vectorInput = screen.getByLabelText('Query Vector');
      fireEvent.change(vectorInput, { target: { value: '0.1, 0.2' } });

      fireEvent.click(screen.getByRole('button', { name: /perform advanced search/i }));

      await waitFor(() => {
        expect(screen.getByText('Result 1')).toBeInTheDocument();
        expect(screen.getByText('Result 2')).toBeInTheDocument();
      });
    });
  });

  describe('Error Handling', () => {
    test('shows alert on search error', async () => {
      searchApi.advancedSearch.mockRejectedValue(new Error('Search failed'));

      render(<AdvancedSearch />);

      await waitFor(() => {
        expect(screen.getByText('Production DB')).toBeInTheDocument();
      });

      fireEvent.change(screen.getByLabelText('Database'), { target: { value: 'db-1' } });

      const vectorInput = screen.getByLabelText('Query Vector');
      fireEvent.change(vectorInput, { target: { value: '0.1, 0.2' } });

      fireEvent.click(screen.getByRole('button', { name: /perform advanced search/i }));

      await waitFor(() => {
        expect(window.alert).toHaveBeenCalledWith(expect.stringContaining('Search failed'));
      });
    });
  });

  describe('Parameter Updates', () => {
    test('updates top k value', async () => {
      render(<AdvancedSearch />);

      await waitFor(() => {
        expect(screen.getByText('Production DB')).toBeInTheDocument();
      });

      fireEvent.change(screen.getByLabelText('Database'), { target: { value: 'db-1' } });

      const topKInput = screen.getByLabelText('Top K Results');
      fireEvent.change(topKInput, { target: { value: '25' } });

      const vectorInput = screen.getByLabelText('Query Vector');
      fireEvent.change(vectorInput, { target: { value: '0.1, 0.2' } });

      fireEvent.click(screen.getByRole('button', { name: /perform advanced search/i }));

      await waitFor(() => {
        expect(searchApi.advancedSearch).toHaveBeenCalledWith('db-1', expect.objectContaining({
          topK: 25
        }));
      });
    });

    test('updates threshold value', async () => {
      render(<AdvancedSearch />);

      await waitFor(() => {
        expect(screen.getByText('Production DB')).toBeInTheDocument();
      });

      fireEvent.change(screen.getByLabelText('Database'), { target: { value: 'db-1' } });

      const thresholdInput = screen.getByLabelText('Similarity Threshold');
      fireEvent.change(thresholdInput, { target: { value: '0.75' } });

      const vectorInput = screen.getByLabelText('Query Vector');
      fireEvent.change(vectorInput, { target: { value: '0.1, 0.2' } });

      fireEvent.click(screen.getByRole('button', { name: /perform advanced search/i }));

      await waitFor(() => {
        expect(searchApi.advancedSearch).toHaveBeenCalledWith('db-1', expect.objectContaining({
          threshold: 0.75
        }));
      });
    });
  });
});
