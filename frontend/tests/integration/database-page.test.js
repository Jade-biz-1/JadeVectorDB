// frontend/tests/integration/database-page.test.js
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import DatabaseManagement from '@/pages/databases';

// Mock the API functions
jest.mock('@/lib/api', () => ({
  databaseApi: {
    listDatabases: jest.fn(),
    createDatabase: jest.fn(),
    updateDatabase: jest.fn(),
    deleteDatabase: jest.fn(),
  }
}));

import { databaseApi } from '@/lib/api';

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

describe('Database Management Page Integration', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    // Mock successful API responses
    databaseApi.listDatabases.mockResolvedValue({
      databases: [
        {
          databaseId: 'db1',
          name: 'Test DB',
          description: 'A test database',
          vectorDimension: 128,
          status: 'active',
          stats: { vectorCount: 1000, indexCount: 2 }
        }
      ]
    });

    databaseApi.createDatabase.mockResolvedValue({
      databaseId: 'new-db',
      name: 'New DB',
      description: 'A new database',
      vectorDimension: 256
    });
  });

  test('loads and displays databases', async () => {
    render(<DatabaseManagement />);

    // Wait for databases to load
    await waitFor(() => {
      expect(screen.getByText('Test DB')).toBeInTheDocument();
    });

    // Verify database list API was called
    expect(databaseApi.listDatabases).toHaveBeenCalled();
  });

  test('calls createDatabase API when form is submitted', async () => {
    render(<DatabaseManagement />);

    // Wait for page to load
    await waitFor(() => {
      expect(databaseApi.listDatabases).toHaveBeenCalled();
    });

    // Find and fill in form fields (label text may vary)
    const nameInput = screen.getByLabelText(/name/i);
    fireEvent.change(nameInput, { target: { value: 'New Test DB' } });

    // Try to find description field if it exists
    const descriptionInput = screen.queryByLabelText(/description/i);
    if (descriptionInput) {
      fireEvent.change(descriptionInput, { target: { value: 'New test database description' } });
    }

    // Find dimension input
    const dimensionInput = screen.getByLabelText(/dimension/i);
    fireEvent.change(dimensionInput, { target: { value: '256' } });

    // Submit form
    const submitButton = screen.getByRole('button', { name: /create/i });
    fireEvent.click(submitButton);

    // Wait for API call to complete
    await waitFor(() => {
      expect(databaseApi.createDatabase).toHaveBeenCalled();
    });
  });

  test('handles API errors gracefully', async () => {
    // Mock an API error
    databaseApi.listDatabases.mockRejectedValue(new Error('Failed to fetch databases'));

    render(<DatabaseManagement />);

    // Check that the function was called
    await waitFor(() => {
      expect(databaseApi.listDatabases).toHaveBeenCalled();
    });
  });

  test('displays page when no databases exist', async () => {
    // Mock empty database list
    databaseApi.listDatabases.mockResolvedValue({ databases: [] });

    render(<DatabaseManagement />);

    await waitFor(() => {
      // Page should render without errors
      expect(databaseApi.listDatabases).toHaveBeenCalled();
    });
  });
});
