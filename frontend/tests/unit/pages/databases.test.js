import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Global mocks: next/head, next/router, next/link via jest.config.js
jest.mock('@/components/Layout', () => {
  return ({ children }) => <div data-testid="layout">{children}</div>;
});
jest.mock('@/lib/api', () => require('./__mocks__/api'));
const { databaseApi } = require('@/lib/api');

import DatabaseManagement from '@/pages/databases';

const mockDatabases = [
  {
    id: 'db-1',
    databaseId: 'db-1',
    name: 'ProductionDB',
    description: 'Main production database',
    vectorDimension: 512,
    status: 'active',
    vectors: 1000,
    indexes: 2,
  },
  {
    id: 'db-2',
    databaseId: 'db-2',
    name: 'TestDB',
    description: 'Test database',
    vectorDimension: 128,
    status: 'inactive',
    vectors: 50,
    indexes: 1,
  },
];

describe('DatabaseManagement page', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    databaseApi.listDatabases.mockResolvedValue({ databases: [] });
  });

  it('renders the page heading', async () => {
    render(<DatabaseManagement />);
    expect(screen.getByText('Database Management')).toBeInTheDocument();
  });

  it('renders the create database form', () => {
    render(<DatabaseManagement />);
    expect(screen.getByText('Create New Database')).toBeInTheDocument();
    expect(screen.getByLabelText(/database name/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/vector dimension/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/index type/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /create database/i })).toBeInTheDocument();
  });

  it('shows empty state when no databases exist', async () => {
    databaseApi.listDatabases.mockResolvedValue({ databases: [] });
    render(<DatabaseManagement />);
    await waitFor(() => {
      expect(screen.getByText(/no databases yet/i)).toBeInTheDocument();
    });
  });

  it('fetches and renders databases on mount', async () => {
    databaseApi.listDatabases.mockResolvedValue({ databases: mockDatabases });
    render(<DatabaseManagement />);
    await waitFor(() => {
      expect(screen.getByText('ProductionDB')).toBeInTheDocument();
      expect(screen.getByText('TestDB')).toBeInTheDocument();
    });
  });

  it('shows database stats (vector count, dimensions)', async () => {
    databaseApi.listDatabases.mockResolvedValue({ databases: mockDatabases });
    render(<DatabaseManagement />);
    await waitFor(() => {
      expect(screen.getByText(/1,000 vectors/i)).toBeInTheDocument();
      expect(screen.getByText(/512D/i)).toBeInTheDocument();
    });
  });

  it('calls createDatabase and refreshes list on form submit', async () => {
    databaseApi.createDatabase.mockResolvedValueOnce({ id: 'new-db' });
    databaseApi.listDatabases
      .mockResolvedValueOnce({ databases: [] })
      .mockResolvedValueOnce({ databases: [{ id: 'new-db', name: 'NewDB', vectorDimension: 128, status: 'active', vectors: 0, indexes: 0 }] });

    render(<DatabaseManagement />);

    await userEvent.type(screen.getByLabelText(/database name/i), 'NewDB');
    fireEvent.click(screen.getByRole('button', { name: /create database/i }));

    await waitFor(() => {
      expect(databaseApi.createDatabase).toHaveBeenCalledWith(
        expect.objectContaining({ name: 'NewDB' })
      );
    });

    await waitFor(() => {
      expect(screen.getByText(/database created successfully/i)).toBeInTheDocument();
    });
  });

  it('shows error when createDatabase fails', async () => {
    databaseApi.createDatabase.mockRejectedValueOnce(new Error('Name already exists'));
    render(<DatabaseManagement />);

    await userEvent.type(screen.getByLabelText(/database name/i), 'DuplicateDB');
    fireEvent.click(screen.getByRole('button', { name: /create database/i }));

    await waitFor(() => {
      expect(screen.getByText(/name already exists/i)).toBeInTheDocument();
    });
  });

  it('shows error when fetching databases fails', async () => {
    databaseApi.listDatabases.mockRejectedValueOnce(new Error('Network error'));
    render(<DatabaseManagement />);
    await waitFor(() => {
      expect(screen.getByText(/error fetching databases/i)).toBeInTheDocument();
    });
  });

  it('disables submit button while creating', async () => {
    databaseApi.createDatabase.mockImplementation(() => new Promise(() => {}));
    render(<DatabaseManagement />);

    await userEvent.type(screen.getByLabelText(/database name/i), 'SlowDB');
    fireEvent.click(screen.getByRole('button', { name: /create database/i }));

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /creating/i })).toBeDisabled();
    });
  });
});
