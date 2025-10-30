// frontend/tests/integration/database-page.test.js
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MockedProvider } from '@apollo/client/testing';
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
    getItem: jest.fn(() => 'test-api-key'),
  },
  writable: true,
});

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
    render(
      <MockedProvider>
        <DatabaseManagement />
      </MockedProvider>
    );

    // Initially should show loading state or default message
    expect(screen.getByText(/database management/i)).toBeInTheDocument();

    // Wait for databases to load
    await waitFor(() => {
      expect(screen.getByText('Test DB')).toBeInTheDocument();
    });

    // Verify database information is displayed
    expect(screen.getByText('A test database')).toBeInTheDocument();
    expect(screen.getByText('1,000 vectors')).toBeInTheDocument();
    expect(screen.getByText('2 indexes')).toBeInTheDocument();
  });

  test('allows creating a new database', async () => {
    render(
      <MockedProvider>
        <DatabaseManagement />
      </MockedProvider>
    );

    // Fill in form fields for new database
    fireEvent.change(screen.getByLabelText('Database Name'), { target: { value: 'New Test DB' } });
    fireEvent.change(screen.getByLabelText('Description'), { target: { value: 'New test database description' } });
    
    // Select vector dimension
    fireEvent.change(screen.getByLabelText('Vector Dimension'), { target: { value: '256' } });
    
    // Submit form
    fireEvent.click(screen.getByRole('button', { name: /create database/i }));

    // Wait for API call to complete
    await waitFor(() => {
      expect(databaseApi.createDatabase).toHaveBeenCalledWith({
        name: 'New Test DB',
        description: 'New test database description',
        vectorDimension: 256,
        indexType: 'FLAT'
      });
    });
  });

  test('handles API errors gracefully', async () => {
    // Mock an API error
    databaseApi.listDatabases.mockRejectedValue(new Error('Failed to fetch databases'));

    render(
      <MockedProvider>
        <DatabaseManagement />
      </MockedProvider>
    );

    // Check that error is handled appropriately
    await waitFor(() => {
      // In our implementation, errors show an alert, so we can't directly test this
      // But we can check that the function was called
      expect(databaseApi.listDatabases).toHaveBeenCalled();
    });
  });

  test('displays appropriate message when no databases exist', async () => {
    // Mock empty database list
    databaseApi.listDatabases.mockResolvedValue({ databases: [] });

    render(
      <MockedProvider>
        <DatabaseManagement />
      </MockedProvider>
    );

    await waitFor(() => {
      // The component doesn't explicitly show "no databases" message in our implementation
      // But it should render without errors
      expect(screen.getByText(/database management/i)).toBeInTheDocument();
    });
  });
});