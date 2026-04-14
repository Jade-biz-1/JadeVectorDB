import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Global mocks: next/head, next/router, next/link via jest.config.js
jest.mock('@/lib/api', () => require('./__mocks__/api'));
const { vectorApi, databaseApi } = require('@/lib/api');

// Mock FileReader
global.FileReader = class {
  onload = null;
  readAsText() {
    if (this.onload) {
      this.onload({ target: { result: JSON.stringify({ vectors: [] }) } });
    }
  }
};

import BatchOperations from '@/pages/batch-operations';

const mockDatabases = [
  { databaseId: 'db-1', name: 'TestDB' },
];

describe('BatchOperations page', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    databaseApi.listDatabases.mockResolvedValue({ databases: mockDatabases });
  });

  it('renders the Batch Vector Operations heading', async () => {
    render(<BatchOperations />);
    expect(screen.getByRole('heading', { name: /batch vector operations/i })).toBeInTheDocument();
  });

  it('loads databases into the selector', async () => {
    render(<BatchOperations />);
    await waitFor(() => {
      expect(screen.getByText('TestDB')).toBeInTheDocument();
    });
  });

  it('shows Upload Vectors button (upload mode is default)', async () => {
    render(<BatchOperations />);
    await waitFor(() => screen.getByText('TestDB'));
    expect(screen.getByRole('button', { name: /upload vectors/i })).toBeInTheDocument();
  });

  it('can switch to download mode via radio button', async () => {
    render(<BatchOperations />);
    await waitFor(() => screen.getByText('TestDB'));

    const downloadRadio = screen.getByRole('radio', { name: /download/i });
    fireEvent.click(downloadRadio);

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /download vectors/i })).toBeInTheDocument();
    });
  });

  it('can add a new vector row', async () => {
    render(<BatchOperations />);
    await waitFor(() => screen.getByText('TestDB'));

    fireEvent.click(screen.getByRole('button', { name: /add vector/i }));

    // Two rows → two Remove buttons
    const removeButtons = screen.getAllByRole('button', { name: /remove/i });
    expect(removeButtons).toHaveLength(2);
  });

  it('calls storeVectorsBatch on upload submit with a selected database', async () => {
    vectorApi.storeVectorsBatch.mockResolvedValueOnce({ count: 1 });
    render(<BatchOperations />);
    await waitFor(() => screen.getByText('TestDB'));

    // Select the database
    const dbSelect = screen.getByRole('combobox');
    fireEvent.change(dbSelect, { target: { value: 'db-1' } });

    // The upload form requires BOTH id and values — use fireEvent.change to set them
    const idInput = screen.getByPlaceholderText('Vector ID');
    const valuesInput = screen.getByPlaceholderText('Comma-separated or JSON array');
    fireEvent.change(idInput, { target: { value: 'test-vec-1' } });
    fireEvent.change(valuesInput, { target: { value: '0.1, 0.2, 0.3' } });

    fireEvent.click(screen.getByRole('button', { name: /upload vectors/i }));

    await waitFor(() => {
      expect(vectorApi.storeVectorsBatch).toHaveBeenCalledWith(
        'db-1',
        expect.arrayContaining([expect.objectContaining({ id: 'test-vec-1' })])
      );
    });
  });
});
