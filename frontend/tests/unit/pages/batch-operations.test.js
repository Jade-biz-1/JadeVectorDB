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

  it('calls storeVector for each vector on upload submit', async () => {
    vectorApi.storeVector.mockResolvedValue({ id: 'test-vec-1' });
    render(<BatchOperations />);
    await waitFor(() => screen.getByText('TestDB'));

    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'db-1' } });

    fireEvent.change(screen.getByPlaceholderText('Vector ID'), { target: { value: 'test-vec-1' } });
    fireEvent.change(screen.getByPlaceholderText('Comma-separated or JSON array'), { target: { value: '0.1, 0.2, 0.3' } });

    fireEvent.click(screen.getByRole('button', { name: /upload vectors/i }));

    await waitFor(() => {
      expect(vectorApi.storeVector).toHaveBeenCalledWith(
        'db-1',
        expect.objectContaining({ id: 'test-vec-1' })
      );
    });
  });

  it('shows success message with vector count after upload', async () => {
    vectorApi.storeVector.mockResolvedValue({});
    render(<BatchOperations />);
    await waitFor(() => screen.getByText('TestDB'));

    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'db-1' } });
    fireEvent.change(screen.getByPlaceholderText('Vector ID'), { target: { value: 'v1' } });
    fireEvent.change(screen.getByPlaceholderText('Comma-separated or JSON array'), { target: { value: '0.1, 0.2' } });

    fireEvent.click(screen.getByRole('button', { name: /upload vectors/i }));

    await waitFor(() => {
      expect(screen.getByText(/Successfully uploaded 1 vector/i)).toBeInTheDocument();
    });
  });

  it('shows partial failure message when some vectors fail', async () => {
    vectorApi.storeVector
      .mockResolvedValueOnce({})
      .mockRejectedValueOnce(new Error('Server error'));

    render(<BatchOperations />);
    await waitFor(() => screen.getByText('TestDB'));

    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'db-1' } });

    // Add two vector rows
    fireEvent.change(screen.getByPlaceholderText('Vector ID'), { target: { value: 'v1' } });
    fireEvent.change(screen.getByPlaceholderText('Comma-separated or JSON array'), { target: { value: '0.1, 0.2' } });

    fireEvent.click(screen.getByRole('button', { name: /add vector/i }));

    const idInputs = screen.getAllByPlaceholderText('Vector ID');
    const valInputs = screen.getAllByPlaceholderText('Comma-separated or JSON array');
    fireEvent.change(idInputs[1], { target: { value: 'v2' } });
    fireEvent.change(valInputs[1], { target: { value: '0.3, 0.4' } });

    fireEvent.click(screen.getByRole('button', { name: /upload vectors/i }));

    await waitFor(() => {
      expect(screen.getByText(/Uploaded 1 vector.*1 failed/i)).toBeInTheDocument();
    });
  });
});
