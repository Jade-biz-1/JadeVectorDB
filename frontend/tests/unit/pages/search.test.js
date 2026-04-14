import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Global mocks: next/head, next/router, next/link via jest.config.js
jest.mock('@/components/Layout', () => {
  return ({ children }) => <div data-testid="layout">{children}</div>;
});
jest.mock('@/lib/api', () => require('./__mocks__/api'));
const { searchApi, databaseApi } = require('@/lib/api');

// Mock window.location.search
Object.defineProperty(window, 'location', {
  writable: true,
  value: { ...window.location, search: '' },
});

import SearchInterface from '@/pages/search';

const mockDatabases = [
  { databaseId: 'db-1', name: 'EmbeddingDB' },
];
const mockResults = [
  { vectorId: 'v-1', score: 0.95, metadata: { title: 'Result A' } },
  { vectorId: 'v-2', score: 0.88, metadata: { title: 'Result B' } },
];

describe('SearchInterface page', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    databaseApi.listDatabases.mockResolvedValue({ databases: mockDatabases });
    window.location.search = '';
  });

  it('renders the Vector Search heading', async () => {
    render(<SearchInterface />);
    // Use heading role to avoid matching button text or other "search" occurrences
    expect(screen.getByRole('heading', { name: /vector search/i })).toBeInTheDocument();
  });

  it('renders the query vector textarea', () => {
    render(<SearchInterface />);
    expect(screen.getByPlaceholderText(/enter vector values/i)).toBeInTheDocument();
  });

  it('loads databases into the selector', async () => {
    render(<SearchInterface />);
    await waitFor(() => {
      expect(screen.getByText('EmbeddingDB')).toBeInTheDocument();
    });
  });

  it('calls searchApi.similaritySearch on form submit', async () => {
    searchApi.similaritySearch.mockResolvedValueOnce({ results: mockResults });
    render(<SearchInterface />);
    await waitFor(() => screen.getByText('EmbeddingDB'));

    // Select a database
    const databaseSelect = screen.getAllByRole('combobox')[0];
    fireEvent.change(databaseSelect, { target: { value: 'db-1' } });

    const vectorInput = screen.getByPlaceholderText(/enter vector values/i);
    await userEvent.type(vectorInput, '0.1, 0.2, 0.3, 0.4');

    // Submit the form directly to avoid HTML required blocking
    const form = vectorInput.closest('form');
    fireEvent.submit(form);

    await waitFor(() => {
      expect(searchApi.similaritySearch).toHaveBeenCalledWith(
        'db-1',
        expect.objectContaining({ queryVector: [0.1, 0.2, 0.3, 0.4] })
      );
    });
  });

  it('displays search results after a successful search', async () => {
    searchApi.similaritySearch.mockResolvedValueOnce({ results: mockResults });
    render(<SearchInterface />);
    await waitFor(() => screen.getByText('EmbeddingDB'));

    const databaseSelect = screen.getAllByRole('combobox')[0];
    fireEvent.change(databaseSelect, { target: { value: 'db-1' } });

    const vectorInput = screen.getByPlaceholderText(/enter vector values/i);
    await userEvent.type(vectorInput, '0.1, 0.2, 0.3, 0.4');
    const form = vectorInput.closest('form');
    fireEvent.submit(form);

    await waitFor(() => {
      // Results show as "Vector ID: v-1"
      expect(screen.getByText(/Vector ID: v-1/i)).toBeInTheDocument();
      expect(screen.getByText(/Vector ID: v-2/i)).toBeInTheDocument();
    });
  });

  it('supports JSON array format for query vector', async () => {
    searchApi.similaritySearch.mockResolvedValueOnce({ results: [] });
    render(<SearchInterface />);
    await waitFor(() => screen.getByText('EmbeddingDB'));

    const databaseSelect = screen.getAllByRole('combobox')[0];
    fireEvent.change(databaseSelect, { target: { value: 'db-1' } });

    const vectorInput = screen.getByPlaceholderText(/enter vector values/i);
    // Use fireEvent.change to avoid userEvent bracket-parsing issues
    fireEvent.change(vectorInput, { target: { value: '[0.1, 0.2, 0.3]' } });
    const form = vectorInput.closest('form');
    fireEvent.submit(form);

    await waitFor(() => {
      expect(searchApi.similaritySearch).toHaveBeenCalledWith(
        'db-1',
        expect.objectContaining({ queryVector: [0.1, 0.2, 0.3] })
      );
    });
  });

  it('shows error message when search fails', async () => {
    searchApi.similaritySearch.mockRejectedValueOnce(new Error('Search failed'));
    render(<SearchInterface />);
    await waitFor(() => screen.getByText('EmbeddingDB'));

    const databaseSelect = screen.getAllByRole('combobox')[0];
    fireEvent.change(databaseSelect, { target: { value: 'db-1' } });

    const vectorInput = screen.getByPlaceholderText(/enter vector values/i);
    await userEvent.type(vectorInput, '0.1, 0.2');
    const form = vectorInput.closest('form');
    fireEvent.submit(form);

    await waitFor(() => {
      expect(screen.getByText(/search failed/i)).toBeInTheDocument();
    });
  });

  it('shows error when database fetch fails', async () => {
    databaseApi.listDatabases.mockRejectedValueOnce(new Error('Network error'));
    render(<SearchInterface />);
    await waitFor(() => {
      expect(screen.getByText(/error fetching databases/i)).toBeInTheDocument();
    });
  });
});
