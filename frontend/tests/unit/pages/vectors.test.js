import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Global mocks: next/head, next/router, next/link via jest.config.js
jest.mock('@/components/Layout', () => {
  return ({ children }) => <div data-testid="layout">{children}</div>;
});
jest.mock('@/lib/api', () => require('./__mocks__/api'));
const { vectorApi, databaseApi } = require('@/lib/api');

// Suppress confirm dialogs
global.confirm = jest.fn(() => false);
global.alert = jest.fn();

import VectorManagement from '@/pages/vectors';

const mockDatabases = [
  { id: 'db-1', databaseId: 'db-1', name: 'TestDB', vectorDimension: 4 },
];
// vectors.js uses vector.id (not vectorId) for display
const mockVectors = [
  { id: 'v-1', values: [0.1, 0.2, 0.3, 0.4], metadata: { label: 'doc1' } },
  { id: 'v-2', values: [0.5, 0.6, 0.7, 0.8], metadata: { label: 'doc2' } },
];

describe('VectorManagement page', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    databaseApi.listDatabases.mockResolvedValue({ databases: mockDatabases });
    databaseApi.getDatabase.mockResolvedValue({ id: 'db-1', name: 'TestDB', vectorDimension: 4 });
    vectorApi.listVectors.mockResolvedValue({ vectors: mockVectors, total: 2 });
  });

  it('renders the page heading', () => {
    render(<VectorManagement />);
    expect(screen.getByText(/vector management/i)).toBeInTheDocument();
  });

  it('fetches and renders database list in selector', async () => {
    render(<VectorManagement />);
    await waitFor(() => {
      expect(databaseApi.listDatabases).toHaveBeenCalled();
      expect(screen.getByText('TestDB')).toBeInTheDocument();
    });
  });

  it('loads vectors when a database is selected', async () => {
    render(<VectorManagement />);
    await waitFor(() => screen.getByText('TestDB'));

    const select = screen.getByRole('combobox');
    fireEvent.change(select, { target: { value: 'db-1' } });

    await waitFor(() => {
      expect(vectorApi.listVectors).toHaveBeenCalledWith('db-1', expect.any(Number), expect.any(Number));
    });

    // Vectors are displayed as "ID: v-1", "ID: v-2"
    await waitFor(() => {
      expect(screen.getByText('ID: v-1')).toBeInTheDocument();
      expect(screen.getByText('ID: v-2')).toBeInTheDocument();
    });
  });

  it('shows empty state when no vectors exist', async () => {
    vectorApi.listVectors.mockResolvedValueOnce({ vectors: [], total: 0 });
    render(<VectorManagement />);
    await waitFor(() => screen.getByText('TestDB'));
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'db-1' } });
    await waitFor(() => {
      expect(screen.getByText(/no vectors/i)).toBeInTheDocument();
    });
  });

  it('shows loading state while fetching vectors', async () => {
    vectorApi.listVectors.mockImplementation(() => new Promise(() => {}));
    render(<VectorManagement />);
    await waitFor(() => screen.getByText('TestDB'));
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'db-1' } });
    await waitFor(() => {
      expect(screen.getByText(/loading/i)).toBeInTheDocument();
    });
  });
});
