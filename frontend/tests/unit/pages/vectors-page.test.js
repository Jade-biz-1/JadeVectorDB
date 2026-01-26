// frontend/tests/unit/pages/vectors-page.test.js
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import VectorManagement from '@/pages/vectors';

// Mock the API functions
jest.mock('@/lib/api', () => ({
  vectorApi: {
    listVectors: jest.fn(),
    storeVector: jest.fn(),
    updateVector: jest.fn(),
    deleteVector: jest.fn(),
  },
  databaseApi: {
    listDatabases: jest.fn(),
    getDatabase: jest.fn(),
  }
}));

import { vectorApi, databaseApi } from '@/lib/api';

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

// Mock window.alert and window.confirm
beforeAll(() => {
  jest.spyOn(window, 'alert').mockImplementation(() => {});
  jest.spyOn(window, 'confirm').mockImplementation(() => true);
});

describe('Vector Management Page', () => {
  const mockDatabases = [
    {
      databaseId: 'db-1',
      id: 'db-1',
      name: 'Test Database',
      vectorDimension: 4,
      indexType: 'HNSW'
    },
    {
      databaseId: 'db-2',
      id: 'db-2',
      name: 'Production DB',
      vectorDimension: 128,
      indexType: 'IVF'
    }
  ];

  const mockVectors = [
    {
      id: 'vec-1',
      values: [0.1, 0.2, 0.3, 0.4],
      metadata: { label: 'test1' }
    },
    {
      id: 'vec-2',
      values: [0.5, 0.6, 0.7, 0.8],
      metadata: { label: 'test2' }
    }
  ];

  beforeEach(() => {
    jest.clearAllMocks();

    // Mock successful API responses
    databaseApi.listDatabases.mockResolvedValue({
      databases: mockDatabases
    });

    databaseApi.getDatabase.mockResolvedValue({
      databaseId: 'db-1',
      name: 'Test Database',
      vectorDimension: 4,
      indexType: 'HNSW'
    });

    vectorApi.listVectors.mockResolvedValue({
      vectors: mockVectors,
      total: 2
    });

    vectorApi.storeVector.mockResolvedValue({ id: 'new-vec' });
    vectorApi.updateVector.mockResolvedValue({ id: 'vec-1' });
    vectorApi.deleteVector.mockResolvedValue({});
  });

  describe('Rendering', () => {
    test('renders page title', async () => {
      render(<VectorManagement />);

      expect(screen.getByText('Vector Management')).toBeInTheDocument();
    });

    test('renders page description', async () => {
      render(<VectorManagement />);

      expect(screen.getByText('Store, manage, and search vector embeddings')).toBeInTheDocument();
    });

    test('renders database selection dropdown', async () => {
      render(<VectorManagement />);

      await waitFor(() => {
        expect(screen.getByText('Select Database')).toBeInTheDocument();
      });

      expect(screen.getByRole('combobox')).toBeInTheDocument();
    });

    test('renders create vector form', async () => {
      render(<VectorManagement />);

      expect(screen.getByText('Create New Vector')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /create vector/i })).toBeInTheDocument();
    });
  });

  describe('Database Selection', () => {
    test('loads and displays databases in dropdown', async () => {
      render(<VectorManagement />);

      await waitFor(() => {
        expect(screen.getByText('Test Database')).toBeInTheDocument();
        expect(screen.getByText('Production DB')).toBeInTheDocument();
      });

      expect(databaseApi.listDatabases).toHaveBeenCalled();
    });

    test('fetches vectors when database is selected', async () => {
      render(<VectorManagement />);

      await waitFor(() => {
        expect(screen.getByText('Test Database')).toBeInTheDocument();
      });

      // Select a database
      const select = screen.getByRole('combobox');
      fireEvent.change(select, { target: { value: 'db-1' } });

      await waitFor(() => {
        expect(vectorApi.listVectors).toHaveBeenCalledWith('db-1', 10, 0);
      });
    });

    test('fetches database details when database is selected', async () => {
      render(<VectorManagement />);

      await waitFor(() => {
        expect(screen.getByText('Test Database')).toBeInTheDocument();
      });

      // Select a database
      const select = screen.getByRole('combobox');
      fireEvent.change(select, { target: { value: 'db-1' } });

      await waitFor(() => {
        expect(databaseApi.getDatabase).toHaveBeenCalledWith('db-1');
      });
    });

    test('shows dimension info when database is selected', async () => {
      render(<VectorManagement />);

      await waitFor(() => {
        expect(screen.getByText('Test Database')).toBeInTheDocument();
      });

      // Select a database
      const select = screen.getByRole('combobox');
      fireEvent.change(select, { target: { value: 'db-1' } });

      await waitFor(() => {
        expect(screen.getByText(/Required Dimension:/)).toBeInTheDocument();
        expect(screen.getByText(/4 values/)).toBeInTheDocument();
      });
    });
  });

  describe('Vector Display', () => {
    test('displays vectors after database selection', async () => {
      render(<VectorManagement />);

      await waitFor(() => {
        expect(screen.getByText('Test Database')).toBeInTheDocument();
      });

      // Select a database
      const select = screen.getByRole('combobox');
      fireEvent.change(select, { target: { value: 'db-1' } });

      await waitFor(() => {
        expect(screen.getByText(/ID: vec-1/)).toBeInTheDocument();
        expect(screen.getByText(/ID: vec-2/)).toBeInTheDocument();
      });
    });

    test('displays vector metadata', async () => {
      render(<VectorManagement />);

      await waitFor(() => {
        expect(screen.getByText('Test Database')).toBeInTheDocument();
      });

      const select = screen.getByRole('combobox');
      fireEvent.change(select, { target: { value: 'db-1' } });

      await waitFor(() => {
        expect(screen.getByText('{"label":"test1"}')).toBeInTheDocument();
      });
    });

    test('shows empty state when no vectors', async () => {
      vectorApi.listVectors.mockResolvedValue({
        vectors: [],
        total: 0
      });

      render(<VectorManagement />);

      await waitFor(() => {
        expect(screen.getByText('Test Database')).toBeInTheDocument();
      });

      const select = screen.getByRole('combobox');
      fireEvent.change(select, { target: { value: 'db-1' } });

      await waitFor(() => {
        expect(screen.getByText('No vectors found in this database')).toBeInTheDocument();
      });
    });

    test('shows prompt to select database when none selected', async () => {
      render(<VectorManagement />);

      await waitFor(() => {
        expect(screen.getByText('Select a database to view vectors')).toBeInTheDocument();
      });
    });
  });

  describe('Create Vector', () => {
    test('create button is disabled without database selection', async () => {
      render(<VectorManagement />);

      const createButton = screen.getByRole('button', { name: /create vector/i });
      expect(createButton).toBeDisabled();
    });

    test('create button is enabled after database selection', async () => {
      render(<VectorManagement />);

      await waitFor(() => {
        expect(screen.getByText('Test Database')).toBeInTheDocument();
      });

      const select = screen.getByRole('combobox');
      fireEvent.change(select, { target: { value: 'db-1' } });

      await waitFor(() => {
        const createButton = screen.getByRole('button', { name: /create vector/i });
        expect(createButton).not.toBeDisabled();
      });
    });

    test('creates vector with correct data', async () => {
      render(<VectorManagement />);

      await waitFor(() => {
        expect(screen.getByText('Test Database')).toBeInTheDocument();
      });

      // Select database
      const select = screen.getByRole('combobox');
      fireEvent.change(select, { target: { value: 'db-1' } });

      await waitFor(() => {
        expect(screen.getByText(/Required Dimension:/)).toBeInTheDocument();
      });

      // Fill in vector values
      const valuesInput = screen.getByPlaceholderText('0.1, 0.2, 0.3, 0.4, ...');
      fireEvent.change(valuesInput, { target: { value: '0.1, 0.2, 0.3, 0.4' } });

      // Fill in metadata
      const metadataInput = screen.getByPlaceholderText('{"label": "example", "category": "test"}');
      fireEvent.change(metadataInput, { target: { value: '{"label": "new"}' } });

      // Submit form
      const createButton = screen.getByRole('button', { name: /create vector/i });
      fireEvent.click(createButton);

      await waitFor(() => {
        expect(vectorApi.storeVector).toHaveBeenCalledWith(
          'db-1',
          expect.objectContaining({
            values: [0.1, 0.2, 0.3, 0.4],
            metadata: { label: 'new' }
          })
        );
      });
    });

    test('shows dimension mismatch alert', async () => {
      render(<VectorManagement />);

      await waitFor(() => {
        expect(screen.getByText('Test Database')).toBeInTheDocument();
      });

      // Select database (expects 4 dimensions)
      const select = screen.getByRole('combobox');
      fireEvent.change(select, { target: { value: 'db-1' } });

      await waitFor(() => {
        expect(screen.getByText(/Required Dimension:/)).toBeInTheDocument();
      });

      // Fill in wrong number of values
      const valuesInput = screen.getByPlaceholderText('0.1, 0.2, 0.3, 0.4, ...');
      fireEvent.change(valuesInput, { target: { value: '0.1, 0.2' } });

      // Submit form
      const createButton = screen.getByRole('button', { name: /create vector/i });
      fireEvent.click(createButton);

      // Should show alert and NOT call API
      expect(window.alert).toHaveBeenCalledWith(expect.stringContaining('Dimension mismatch'));
      expect(vectorApi.storeVector).not.toHaveBeenCalled();
    });
  });

  describe('Edit Vector', () => {
    test('opens edit modal when edit button is clicked', async () => {
      render(<VectorManagement />);

      await waitFor(() => {
        expect(screen.getByText('Test Database')).toBeInTheDocument();
      });

      const select = screen.getByRole('combobox');
      fireEvent.change(select, { target: { value: 'db-1' } });

      await waitFor(() => {
        expect(screen.getByText(/ID: vec-1/)).toBeInTheDocument();
      });

      // Click edit button
      const editButtons = screen.getAllByRole('button', { name: /edit/i });
      fireEvent.click(editButtons[0]);

      // Modal should be open
      await waitFor(() => {
        expect(screen.getByText('Edit Vector')).toBeInTheDocument();
      });
    });

    test('updates vector when form is submitted', async () => {
      render(<VectorManagement />);

      await waitFor(() => {
        expect(screen.getByText('Test Database')).toBeInTheDocument();
      });

      const select = screen.getByRole('combobox');
      fireEvent.change(select, { target: { value: 'db-1' } });

      await waitFor(() => {
        expect(screen.getByText(/ID: vec-1/)).toBeInTheDocument();
      });

      // Click edit button
      const editButtons = screen.getAllByRole('button', { name: /edit/i });
      fireEvent.click(editButtons[0]);

      await waitFor(() => {
        expect(screen.getByText('Edit Vector')).toBeInTheDocument();
      });

      // Submit the update form
      const updateButton = screen.getByRole('button', { name: /update vector/i });
      fireEvent.click(updateButton);

      await waitFor(() => {
        expect(vectorApi.updateVector).toHaveBeenCalledWith(
          'db-1',
          'vec-1',
          expect.objectContaining({
            values: expect.any(Array),
            metadata: expect.any(Object)
          })
        );
      });
    });

    test('closes modal when cancel button is clicked', async () => {
      render(<VectorManagement />);

      await waitFor(() => {
        expect(screen.getByText('Test Database')).toBeInTheDocument();
      });

      const select = screen.getByRole('combobox');
      fireEvent.change(select, { target: { value: 'db-1' } });

      await waitFor(() => {
        expect(screen.getByText(/ID: vec-1/)).toBeInTheDocument();
      });

      // Click edit button
      const editButtons = screen.getAllByRole('button', { name: /edit/i });
      fireEvent.click(editButtons[0]);

      await waitFor(() => {
        expect(screen.getByText('Edit Vector')).toBeInTheDocument();
      });

      // Click cancel
      const cancelButton = screen.getByRole('button', { name: /cancel/i });
      fireEvent.click(cancelButton);

      await waitFor(() => {
        expect(screen.queryByText('Edit Vector')).not.toBeInTheDocument();
      });
    });
  });

  describe('Delete Vector', () => {
    test('deletes vector when delete button is clicked and confirmed', async () => {
      render(<VectorManagement />);

      await waitFor(() => {
        expect(screen.getByText('Test Database')).toBeInTheDocument();
      });

      const select = screen.getByRole('combobox');
      fireEvent.change(select, { target: { value: 'db-1' } });

      await waitFor(() => {
        expect(screen.getByText(/ID: vec-1/)).toBeInTheDocument();
      });

      // Click delete button
      const deleteButtons = screen.getAllByRole('button', { name: /delete/i });
      fireEvent.click(deleteButtons[0]);

      await waitFor(() => {
        expect(vectorApi.deleteVector).toHaveBeenCalledWith('db-1', 'vec-1');
      });
    });

    test('does not delete when confirmation is cancelled', async () => {
      window.confirm.mockReturnValueOnce(false);

      render(<VectorManagement />);

      await waitFor(() => {
        expect(screen.getByText('Test Database')).toBeInTheDocument();
      });

      const select = screen.getByRole('combobox');
      fireEvent.change(select, { target: { value: 'db-1' } });

      await waitFor(() => {
        expect(screen.getByText(/ID: vec-1/)).toBeInTheDocument();
      });

      // Click delete button
      const deleteButtons = screen.getAllByRole('button', { name: /delete/i });
      fireEvent.click(deleteButtons[0]);

      // Should NOT call delete API
      expect(vectorApi.deleteVector).not.toHaveBeenCalled();
    });
  });

  describe('Pagination', () => {
    test('shows pagination when multiple pages exist', async () => {
      vectorApi.listVectors.mockResolvedValue({
        vectors: mockVectors,
        total: 25 // More than one page
      });

      render(<VectorManagement />);

      await waitFor(() => {
        expect(screen.getByText('Test Database')).toBeInTheDocument();
      });

      const select = screen.getByRole('combobox');
      fireEvent.change(select, { target: { value: 'db-1' } });

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /previous/i })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /next/i })).toBeInTheDocument();
      });
    });

    test('previous button is disabled on first page', async () => {
      vectorApi.listVectors.mockResolvedValue({
        vectors: mockVectors,
        total: 25
      });

      render(<VectorManagement />);

      await waitFor(() => {
        expect(screen.getByText('Test Database')).toBeInTheDocument();
      });

      const select = screen.getByRole('combobox');
      fireEvent.change(select, { target: { value: 'db-1' } });

      await waitFor(() => {
        const prevButton = screen.getByRole('button', { name: /previous/i });
        expect(prevButton).toBeDisabled();
      });
    });

    test('navigates to next page when next button is clicked', async () => {
      vectorApi.listVectors.mockResolvedValue({
        vectors: mockVectors,
        total: 25
      });

      render(<VectorManagement />);

      await waitFor(() => {
        expect(screen.getByText('Test Database')).toBeInTheDocument();
      });

      const select = screen.getByRole('combobox');
      fireEvent.change(select, { target: { value: 'db-1' } });

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /next/i })).toBeInTheDocument();
      });

      // Clear mocks to track new calls
      jest.clearAllMocks();

      // Click next
      const nextButton = screen.getByRole('button', { name: /next/i });
      fireEvent.click(nextButton);

      await waitFor(() => {
        expect(vectorApi.listVectors).toHaveBeenCalledWith('db-1', 10, 10);
      });
    });

    test('shows vector count information', async () => {
      vectorApi.listVectors.mockResolvedValue({
        vectors: mockVectors,
        total: 25
      });

      render(<VectorManagement />);

      await waitFor(() => {
        expect(screen.getByText('Test Database')).toBeInTheDocument();
      });

      const select = screen.getByRole('combobox');
      fireEvent.change(select, { target: { value: 'db-1' } });

      // Text shows pageSize range (1-10) not actual vector count (1-2)
      await waitFor(() => {
        expect(screen.getByText(/Showing 1-10 of 25 vectors/)).toBeInTheDocument();
      });
    });
  });

  describe('Error Handling', () => {
    test('handles database fetch error', async () => {
      databaseApi.listDatabases.mockRejectedValue(new Error('Network error'));

      render(<VectorManagement />);

      await waitFor(() => {
        expect(window.alert).toHaveBeenCalledWith(expect.stringContaining('Error fetching databases'));
      });
    });

    test('handles vector fetch error', async () => {
      vectorApi.listVectors.mockRejectedValue(new Error('Network error'));

      render(<VectorManagement />);

      await waitFor(() => {
        expect(screen.getByText('Test Database')).toBeInTheDocument();
      });

      const select = screen.getByRole('combobox');
      fireEvent.change(select, { target: { value: 'db-1' } });

      await waitFor(() => {
        expect(window.alert).toHaveBeenCalledWith(expect.stringContaining('Error fetching vectors'));
      });
    });

    test('handles create vector error', async () => {
      vectorApi.storeVector.mockRejectedValue(new Error('Failed to create'));

      render(<VectorManagement />);

      await waitFor(() => {
        expect(screen.getByText('Test Database')).toBeInTheDocument();
      });

      const select = screen.getByRole('combobox');
      fireEvent.change(select, { target: { value: 'db-1' } });

      await waitFor(() => {
        expect(screen.getByText(/Required Dimension:/)).toBeInTheDocument();
      });

      // Fill in vector values
      const valuesInput = screen.getByPlaceholderText('0.1, 0.2, 0.3, 0.4, ...');
      fireEvent.change(valuesInput, { target: { value: '0.1, 0.2, 0.3, 0.4' } });

      // Submit form
      const createButton = screen.getByRole('button', { name: /create vector/i });
      fireEvent.click(createButton);

      await waitFor(() => {
        expect(window.alert).toHaveBeenCalledWith(expect.stringContaining('Error creating vector'));
      });
    });
  });
});
