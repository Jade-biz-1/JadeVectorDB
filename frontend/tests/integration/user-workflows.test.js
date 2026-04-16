// frontend/tests/integration/user-workflows.test.js
// Integration tests for complete user workflows spanning page interactions.
// Covers: Database Management, Vector Management, Similarity Search,
//         Batch Upload, User Management, Alerting.

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';

// ─── Global mocks (next/head, router, link) come from jest.config.js ────────

// ─── Mock @/lib/api with all services needed across workflows ────────────────
jest.mock('@/lib/api', () => ({
  databaseApi: {
    listDatabases:  jest.fn(),
    createDatabase: jest.fn(),
    getDatabase:    jest.fn(),
    deleteDatabase: jest.fn(),
    updateDatabase: jest.fn(),
  },
  vectorApi: {
    listVectors:       jest.fn(),
    storeVector:       jest.fn(),
    storeVectorsBatch: jest.fn(),
    getVector:         jest.fn(),
    deleteVector:      jest.fn(),
    updateVector:      jest.fn(),
  },
  searchApi: {
    similaritySearch: jest.fn(),
  },
  usersApi: {
    listUsers:   jest.fn(),
    createUser:  jest.fn(),
    deleteUser:  jest.fn(),
    updateUser:  jest.fn(),
  },
  alertApi: {
    listAlerts:        jest.fn(),
    acknowledgeAlert:  jest.fn(),
  },
  monitoringApi: {
    getSystemHealth: jest.fn(),
    getSystemStats:  jest.fn(),
  },
  authApi: {
    login:           jest.fn(),
    logout:          jest.fn(),
    register:        jest.fn(),
    changePassword:  jest.fn(),
    forgotPassword:  jest.fn(),
    resetPassword:   jest.fn(),
  },
  apiKeysApi: {
    listApiKeys:   jest.fn(),
    createApiKey:  jest.fn(),
    deleteApiKey:  jest.fn(),
  },
  clusterApi: {
    getClusterStatus: jest.fn(),
    getNodes:         jest.fn(),
  },
  securityApi: {
    getSecurityLogs:   jest.fn(),
    getSecurityStatus: jest.fn(),
  },
  performanceApi: {
    getMetrics: jest.fn(),
  },
}));

// Import after mock declaration
import {
  databaseApi, vectorApi, searchApi, usersApi, alertApi,
} from '@/lib/api';

import DatabaseManagement from '@/pages/databases';
import VectorManagement   from '@/pages/vectors';
import SearchInterface    from '@/pages/search';
import BatchOperations    from '@/pages/batch-operations';
import UserManagement     from '@/pages/users';
import Alerting           from '@/pages/alerting';

// ─── Shared fixtures ──────────────────────────────────────────────────────────
const DATABASES = [
  { databaseId: 'db-alpha', name: 'AlphaDB', vectorDimension: 4,  status: 'active', stats: { vectorCount: 3 } },
  { databaseId: 'db-beta',  name: 'BetaDB',  vectorDimension: 8,  status: 'active', stats: { vectorCount: 0 } },
];
const VECTORS = [
  { id: 'v-1', values: [0.1, 0.2, 0.3, 0.4], metadata: { label: 'doc-A' } },
  { id: 'v-2', values: [0.5, 0.6, 0.7, 0.8], metadata: { label: 'doc-B' } },
  { id: 'v-3', values: [0.9, 0.8, 0.7, 0.6], metadata: { label: 'doc-C' } },
];
const SEARCH_RESULTS = [
  { vectorId: 'v-1', score: 0.98, metadata: { label: 'doc-A' } },
  { vectorId: 'v-3', score: 0.82, metadata: { label: 'doc-C' } },
];
const USERS = [
  { user_id: 'u-1', username: 'alice', email: 'alice@corp.com', roles: ['admin'],  active: true },
  { user_id: 'u-2', username: 'bob',   email: 'bob@corp.com',   roles: ['viewer'], active: true },
];
const ALERTS = [
  { id: 'a-1', type: 'error',   message: 'Disk space critical', timestamp: '2026-04-13T10:00:00Z' },
  { id: 'a-2', type: 'warning', message: 'Memory high',         timestamp: '2026-04-13T10:01:00Z' },
  { id: 'a-3', type: 'info',    message: 'Backup completed',    timestamp: '2026-04-13T10:02:00Z' },
];

// ─────────────────────────────────────────────────────────────────────────────
// Workflow 1: Database Management
// ─────────────────────────────────────────────────────────────────────────────
describe('Workflow: Database Management', () => {
  beforeEach(() => jest.clearAllMocks());

  it('displays all databases returned by the API', async () => {
    databaseApi.listDatabases.mockResolvedValue({ databases: DATABASES });

    render(<DatabaseManagement />);
    await waitFor(() => {
      expect(screen.getByText('AlphaDB')).toBeInTheDocument();
      expect(screen.getByText('BetaDB')).toBeInTheDocument();
    });
  });

  it('creates a database and shows a success message', async () => {
    databaseApi.listDatabases
      .mockResolvedValueOnce({ databases: [] })
      .mockResolvedValueOnce({ databases: [{ databaseId: 'db-new', name: 'NewDB', vectorDimension: 128, stats: { vectorCount: 0 } }] });
    databaseApi.createDatabase.mockResolvedValueOnce({ databaseId: 'db-new', name: 'NewDB' });

    render(<DatabaseManagement />);
    await waitFor(() => expect(databaseApi.listDatabases).toHaveBeenCalled());

    fireEvent.change(screen.getByLabelText(/name/i),      { target: { value: 'NewDB' } });
    fireEvent.change(screen.getByLabelText(/dimension/i), { target: { value: '128'  } });

    fireEvent.submit(screen.getByLabelText(/name/i).closest('form'));

    await waitFor(() => {
      expect(databaseApi.createDatabase).toHaveBeenCalledWith(
        expect.objectContaining({ name: 'NewDB', vectorDimension: 128 })
      );
    });
    await waitFor(() => {
      expect(screen.getByText(/database created successfully/i)).toBeInTheDocument();
    });
  });

  it('shows an error when database creation fails', async () => {
    databaseApi.listDatabases.mockResolvedValue({ databases: DATABASES });
    databaseApi.createDatabase.mockRejectedValueOnce(new Error('Name already taken'));

    render(<DatabaseManagement />);
    await waitFor(() => screen.getByText('AlphaDB'));

    fireEvent.change(screen.getByLabelText(/name/i),      { target: { value: 'AlphaDB' } });
    fireEvent.change(screen.getByLabelText(/dimension/i), { target: { value: '4' } });

    fireEvent.submit(screen.getByLabelText(/name/i).closest('form'));

    await waitFor(() => {
      expect(screen.getByText(/name already taken/i)).toBeInTheDocument();
    });
  });

  it('shows an error when database fetch fails on mount', async () => {
    databaseApi.listDatabases.mockRejectedValueOnce(new Error('Service unavailable'));

    render(<DatabaseManagement />);
    await waitFor(() => {
      expect(screen.getByText(/error fetching databases/i)).toBeInTheDocument();
    });
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Workflow 2: Vector Management (select DB → load vectors → view)
// ─────────────────────────────────────────────────────────────────────────────
describe('Workflow: Vector Management', () => {
  beforeEach(() => jest.clearAllMocks());

  it('selects a database and loads its vectors', async () => {
    databaseApi.listDatabases.mockResolvedValue({ databases: DATABASES });
    databaseApi.getDatabase.mockResolvedValue({ id: 'db-alpha', name: 'AlphaDB', vectorDimension: 4 });
    vectorApi.listVectors.mockResolvedValue({ vectors: VECTORS, total: VECTORS.length });

    render(<VectorManagement />);
    await waitFor(() => screen.getByText('AlphaDB'));

    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'db-alpha' } });

    await waitFor(() => {
      expect(vectorApi.listVectors).toHaveBeenCalledWith('db-alpha', expect.any(Number), expect.any(Number));
    });
    await waitFor(() => {
      expect(screen.getByText('ID: v-1')).toBeInTheDocument();
      expect(screen.getByText('ID: v-2')).toBeInTheDocument();
      expect(screen.getByText('ID: v-3')).toBeInTheDocument();
    });
  });

  it('shows empty state when the selected database has no vectors', async () => {
    databaseApi.listDatabases.mockResolvedValue({ databases: DATABASES });
    databaseApi.getDatabase.mockResolvedValue({ id: 'db-beta', name: 'BetaDB', vectorDimension: 8 });
    vectorApi.listVectors.mockResolvedValue({ vectors: [], total: 0 });

    render(<VectorManagement />);
    await waitFor(() => screen.getByText('BetaDB'));

    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'db-beta' } });

    await waitFor(() => {
      expect(screen.getByText(/no vectors/i)).toBeInTheDocument();
    });
  });

  it('shows a loading indicator while fetching vectors', async () => {
    databaseApi.listDatabases.mockResolvedValue({ databases: DATABASES });
    databaseApi.getDatabase.mockResolvedValue({ id: 'db-alpha', name: 'AlphaDB', vectorDimension: 4 });
    vectorApi.listVectors.mockImplementation(() => new Promise(() => {})); // never resolves

    render(<VectorManagement />);
    await waitFor(() => screen.getByText('AlphaDB'));

    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'db-alpha' } });

    await waitFor(() => {
      expect(screen.getByText(/loading/i)).toBeInTheDocument();
    });
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Workflow 3: Similarity Search (select DB → enter vector → view results)
// ─────────────────────────────────────────────────────────────────────────────
describe('Workflow: Similarity Search', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    window.location.search = '';
  });

  it('performs a search and displays ranked results', async () => {
    databaseApi.listDatabases.mockResolvedValue({ databases: DATABASES });
    searchApi.similaritySearch.mockResolvedValueOnce({ results: SEARCH_RESULTS });

    render(<SearchInterface />);
    await waitFor(() => screen.getByText('AlphaDB'));

    fireEvent.change(screen.getAllByRole('combobox')[0], { target: { value: 'db-alpha' } });

    const vectorInput = screen.getByPlaceholderText(/enter vector values/i);
    fireEvent.change(vectorInput, { target: { value: '0.1, 0.2, 0.3, 0.4' } });

    fireEvent.submit(vectorInput.closest('form'));

    await waitFor(() => {
      expect(searchApi.similaritySearch).toHaveBeenCalledWith(
        'db-alpha',
        expect.objectContaining({ queryVector: [0.1, 0.2, 0.3, 0.4] })
      );
    });
    await waitFor(() => {
      expect(screen.getByText(/Vector ID: v-1/i)).toBeInTheDocument();
      expect(screen.getByText(/Vector ID: v-3/i)).toBeInTheDocument();
    });
  });

  it('accepts JSON array format for the query vector', async () => {
    databaseApi.listDatabases.mockResolvedValue({ databases: DATABASES });
    searchApi.similaritySearch.mockResolvedValueOnce({ results: [] });

    render(<SearchInterface />);
    await waitFor(() => screen.getByText('AlphaDB'));

    fireEvent.change(screen.getAllByRole('combobox')[0], { target: { value: 'db-alpha' } });

    const vectorInput = screen.getByPlaceholderText(/enter vector values/i);
    fireEvent.change(vectorInput, { target: { value: '[0.5, 0.6, 0.7, 0.8]' } });

    fireEvent.submit(vectorInput.closest('form'));

    await waitFor(() => {
      expect(searchApi.similaritySearch).toHaveBeenCalledWith(
        'db-alpha',
        expect.objectContaining({ queryVector: [0.5, 0.6, 0.7, 0.8] })
      );
    });
  });

  it('shows an error message when the search API fails', async () => {
    databaseApi.listDatabases.mockResolvedValue({ databases: DATABASES });
    searchApi.similaritySearch.mockRejectedValueOnce(new Error('Index not ready'));

    render(<SearchInterface />);
    await waitFor(() => screen.getByText('AlphaDB'));

    fireEvent.change(screen.getAllByRole('combobox')[0], { target: { value: 'db-alpha' } });

    const vectorInput = screen.getByPlaceholderText(/enter vector values/i);
    fireEvent.change(vectorInput, { target: { value: '0.1, 0.2, 0.3, 0.4' } });

    fireEvent.submit(vectorInput.closest('form'));

    await waitFor(() => {
      expect(screen.getByText(/index not ready/i)).toBeInTheDocument();
    });
  });

  it('shows an error when the database list fails to load', async () => {
    databaseApi.listDatabases.mockRejectedValueOnce(new Error('Service unavailable'));

    render(<SearchInterface />);

    await waitFor(() => {
      expect(screen.getByText(/error fetching databases/i)).toBeInTheDocument();
    });
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Workflow 4: Batch Vector Upload
// ─────────────────────────────────────────────────────────────────────────────
describe('Workflow: Batch Vector Upload', () => {
  beforeEach(() => jest.clearAllMocks());

  it('uploads vectors to the selected database', async () => {
    databaseApi.listDatabases.mockResolvedValue({ databases: DATABASES });
    vectorApi.storeVector.mockResolvedValue({ id: 'batch-v-1' });

    render(<BatchOperations />);
    await waitFor(() => screen.getByText('AlphaDB'));

    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'db-alpha' } });

    fireEvent.change(screen.getByPlaceholderText('Vector ID'),
      { target: { value: 'batch-v-1' } });
    fireEvent.change(screen.getByPlaceholderText('Comma-separated or JSON array'),
      { target: { value: '0.1, 0.2, 0.3, 0.4' } });

    fireEvent.click(screen.getByRole('button', { name: /upload vectors/i }));

    await waitFor(() => {
      expect(vectorApi.storeVector).toHaveBeenCalledWith(
        'db-alpha',
        expect.objectContaining({ id: 'batch-v-1' })
      );
    });
  });

  it('switches to download mode and shows the download button', async () => {
    databaseApi.listDatabases.mockResolvedValue({ databases: DATABASES });

    render(<BatchOperations />);
    await waitFor(() => screen.getByText('AlphaDB'));

    fireEvent.click(screen.getByRole('radio', { name: /download/i }));

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /download vectors/i })).toBeInTheDocument();
    });
  });

  it('adds a second vector row when Add Vector is clicked', async () => {
    databaseApi.listDatabases.mockResolvedValue({ databases: DATABASES });

    render(<BatchOperations />);
    await waitFor(() => screen.getByText('AlphaDB'));

    fireEvent.click(screen.getByRole('button', { name: /add vector/i }));

    expect(screen.getAllByRole('button', { name: /remove/i })).toHaveLength(2);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Workflow 5: User Management
// ─────────────────────────────────────────────────────────────────────────────
describe('Workflow: User Management', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    global.confirm = jest.fn(() => false);
  });

  it('loads and displays users on mount', async () => {
    usersApi.listUsers.mockResolvedValue({ users: USERS });

    render(<UserManagement />);
    await waitFor(() => {
      expect(screen.getByText('alice')).toBeInTheDocument();
      expect(screen.getByText('bob')).toBeInTheDocument();
      expect(screen.getByText('alice@corp.com')).toBeInTheDocument();
    });
  });

  it('creates a new user and shows a success message', async () => {
    usersApi.listUsers.mockResolvedValue({ users: USERS });
    usersApi.createUser.mockResolvedValueOnce({ user_id: 'u-3' });

    render(<UserManagement />);
    await waitFor(() => screen.getByText('alice'));

    fireEvent.change(screen.getByPlaceholderText('john_doe'),         { target: { value: 'charlie' } });
    fireEvent.change(screen.getByPlaceholderText('john@example.com'), { target: { value: 'charlie@corp.com' } });
    fireEvent.change(screen.getByPlaceholderText('Enter password'),   { target: { value: 'Secure123!' } });

    fireEvent.submit(screen.getByPlaceholderText('john_doe').closest('form'));

    await waitFor(() => {
      expect(usersApi.createUser).toHaveBeenCalledWith(
        'charlie', 'Secure123!', 'charlie@corp.com', expect.any(Array)
      );
    });
    await waitFor(() => {
      expect(screen.getByText(/user created successfully/i)).toBeInTheDocument();
    });
  });

  it('shows an error when createUser fails due to duplicate username', async () => {
    usersApi.listUsers.mockResolvedValue({ users: USERS });
    usersApi.createUser.mockRejectedValueOnce(new Error('Username already exists'));

    render(<UserManagement />);
    await waitFor(() => screen.getByText('alice'));

    fireEvent.change(screen.getByPlaceholderText('john_doe'),         { target: { value: 'alice' } });
    fireEvent.change(screen.getByPlaceholderText('john@example.com'), { target: { value: 'alice2@corp.com' } });
    fireEvent.change(screen.getByPlaceholderText('Enter password'),   { target: { value: 'Secure123!' } });

    fireEvent.submit(screen.getByPlaceholderText('john_doe').closest('form'));

    await waitFor(() => {
      expect(screen.getByText(/username already exists/i)).toBeInTheDocument();
    });
  });

  it('shows an error when listing users fails', async () => {
    usersApi.listUsers.mockRejectedValueOnce(new Error('Forbidden'));

    render(<UserManagement />);
    await waitFor(() => {
      expect(screen.getByText(/error fetching users/i)).toBeInTheDocument();
    });
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Workflow 6: Alerting (fetch → filter → auto-refresh)
// ─────────────────────────────────────────────────────────────────────────────
describe('Workflow: Alerting', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    jest.clearAllMocks();
    alertApi.listAlerts.mockResolvedValue({ alerts: ALERTS });
  });

  afterEach(() => jest.useRealTimers());

  it('fetches and displays all alerts on mount', async () => {
    await act(async () => render(<Alerting />));
    await waitFor(() => {
      expect(screen.getByText(/disk space critical/i)).toBeInTheDocument();
      expect(screen.getByText(/memory high/i)).toBeInTheDocument();
      expect(screen.getByText(/backup completed/i)).toBeInTheDocument();
    });
  });

  it('shows the correct alert count in the heading', async () => {
    await act(async () => render(<Alerting />));
    await waitFor(() => {
      expect(screen.getByText(/\(3\)/)).toBeInTheDocument();
    });
  });

  it('filters to show only error-type alerts', async () => {
    await act(async () => render(<Alerting />));
    await waitFor(() => screen.getByText(/disk space critical/i));

    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'error' } });

    expect(screen.getByText(/disk space critical/i)).toBeInTheDocument();
    expect(screen.queryByText(/memory high/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/backup completed/i)).not.toBeInTheDocument();
  });

  it('shows all alerts when the filter is reset to "all"', async () => {
    await act(async () => render(<Alerting />));
    await waitFor(() => screen.getByText(/disk space critical/i));

    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'error' } });
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'all'   } });

    expect(screen.getByText(/disk space critical/i)).toBeInTheDocument();
    expect(screen.getByText(/memory high/i)).toBeInTheDocument();
    expect(screen.getByText(/backup completed/i)).toBeInTheDocument();
  });

  it('auto-refreshes alerts every 30 seconds', async () => {
    await act(async () => render(<Alerting />));
    await waitFor(() => expect(alertApi.listAlerts).toHaveBeenCalledTimes(1));

    await act(async () => { jest.advanceTimersByTime(30000); });
    await waitFor(() => expect(alertApi.listAlerts).toHaveBeenCalledTimes(2));
  });

  it('clears the refresh interval on unmount', async () => {
    const clearIntervalSpy = jest.spyOn(global, 'clearInterval');
    const { unmount } = await act(async () => render(<Alerting />));
    unmount();
    expect(clearIntervalSpy).toHaveBeenCalled();
    clearIntervalSpy.mockRestore();
  });
});
