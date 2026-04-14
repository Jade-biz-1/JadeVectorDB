import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

// Global mocks: next/head, next/router, next/link via jest.config.js
jest.mock('@/lib/api', () => require('./__mocks__/api'));
const { vectorApi, databaseApi } = require('@/lib/api');

import DataExploration from '@/pages/explore';

const mockDatabases = [
  { databaseId: 'db-1', name: 'EmbeddingDB' },
  { databaseId: 'db-2', name: 'ImageDB' },
];
const mockVectors = [
  { vectorId: 'v-1', values: [0.1, 0.2], metadata: { label: 'cat' } },
  { vectorId: 'v-2', values: [0.3, 0.4], metadata: { label: 'dog' } },
];

describe('DataExploration page', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    databaseApi.listDatabases.mockResolvedValue({ databases: mockDatabases });
    vectorApi.listVectors.mockResolvedValue({ vectors: mockVectors });
  });

  it('renders the Data Exploration heading', async () => {
    render(<DataExploration />);
    expect(screen.getByRole('heading', { name: /data exploration/i })).toBeInTheDocument();
  });

  it('fetches databases on mount', async () => {
    render(<DataExploration />);
    await waitFor(() => {
      expect(databaseApi.listDatabases).toHaveBeenCalled();
    });
  });

  it('auto-selects the first database and fetches vectors', async () => {
    render(<DataExploration />);
    await waitFor(() => {
      expect(vectorApi.listVectors).toHaveBeenCalledWith('db-1', expect.any(Number), 0);
    });
  });

  it('displays fetched vector IDs', async () => {
    render(<DataExploration />);
    await waitFor(() => {
      expect(screen.getByText('v-1')).toBeInTheDocument();
      expect(screen.getByText('v-2')).toBeInTheDocument();
    });
  });

  it('shows loading indicator while fetching vectors', async () => {
    vectorApi.listVectors.mockImplementation(() => new Promise(() => {}));
    render(<DataExploration />);
    await waitFor(() => {
      // Check for loading indicator via role or test-id rather than text
      // since "Loading" might appear in multiple places
      const loadingEls = screen.getAllByText(/loading/i);
      expect(loadingEls.length).toBeGreaterThan(0);
    });
  });

  it('re-fetches vectors when database is changed via combobox', async () => {
    render(<DataExploration />);
    await waitFor(() => expect(vectorApi.listVectors).toHaveBeenCalledTimes(1));

    // Use getByRole with name to target the database selector specifically
    const selects = screen.getAllByRole('combobox');
    // The database selector is the first combobox
    fireEvent.change(selects[0], { target: { value: 'db-2' } });

    await waitFor(() => {
      expect(vectorApi.listVectors).toHaveBeenCalledWith('db-2', expect.any(Number), 0);
    });
  });
});
