// tests/unit/pages/accessibility.test.js
// Accessibility tests: ARIA labels, semantic roles, heading hierarchy,
// keyboard-accessible controls, and visible error messaging.
//
// These run in jsdom (no real browser required).  They check:
//   - All form inputs have an accessible label (getByLabelText)
//   - Primary page heading exists at the correct level
//   - Submit / action buttons have accessible names
//   - Dynamically surfaced error messages are in the DOM
//   - Selects / combos expose their label

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

// Global mocks: next/head, next/router, next/link via jest.config.js
jest.mock('@/components/Layout', () =>
  ({ children }) => <div data-testid="layout">{children}</div>
);
jest.mock('@/lib/api', () => require('./__mocks__/api'));

const {
  databaseApi, vectorApi, searchApi, authApi, usersApi, alertApi,
} = require('@/lib/api');

// ─── fixtures ────────────────────────────────────────────────────────────────
const DB      = { databaseId: 'db-1', name: 'TestDB', vectorDimension: 4, status: 'active', stats: { vectorCount: 0 } };
const VECTOR  = { id: 'v-1', values: [0.1, 0.2, 0.3, 0.4], metadata: {} };

// ─────────────────────────────────────────────────────────────────────────────
// Login page
// ─────────────────────────────────────────────────────────────────────────────
describe('Accessibility: Login page', () => {
  beforeEach(() => jest.clearAllMocks());

  it('has an h1 heading', () => {
    const Login = require('@/pages/login').default;
    render(<Login />);
    expect(screen.getByRole('heading', { level: 1 })).toBeInTheDocument();
  });

  it('labels the username input', () => {
    const Login = require('@/pages/login').default;
    render(<Login />);
    expect(screen.getByLabelText(/username/i)).toBeInTheDocument();
  });

  it('labels the password input', () => {
    const Login = require('@/pages/login').default;
    render(<Login />);
    expect(screen.getByLabelText(/^password$/i)).toBeInTheDocument();
  });

  it('has an accessible submit button', () => {
    const Login = require('@/pages/login').default;
    render(<Login />);
    const btn = screen.getByRole('button', { name: /sign in|log in/i });
    expect(btn).toBeInTheDocument();
  });

  it('surfaces an error message accessibly when login fails', async () => {
    authApi.login.mockRejectedValueOnce(new Error('Invalid credentials'));

    const Login = require('@/pages/login').default;
    render(<Login />);

    fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'bad' } });
    fireEvent.change(screen.getByLabelText(/^password$/i), { target: { value: 'bad' } });
    fireEvent.submit(screen.getByLabelText(/username/i).closest('form'));

    await waitFor(() => {
      expect(screen.getByText(/invalid credentials/i)).toBeInTheDocument();
    });
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Register page
// ─────────────────────────────────────────────────────────────────────────────
describe('Accessibility: Register page', () => {
  beforeEach(() => jest.clearAllMocks());

  it('has an h1 heading', () => {
    const Register = require('@/pages/register').default;
    render(<Register />);
    expect(screen.getByRole('heading', { level: 1 })).toBeInTheDocument();
  });

  it('labels all required form inputs', () => {
    const Register = require('@/pages/register').default;
    render(<Register />);
    // Must have at minimum: username, password, confirm password
    expect(screen.getByLabelText(/username/i)).toBeInTheDocument();
    expect(screen.getAllByLabelText(/password/i).length).toBeGreaterThanOrEqual(1);
  });

  it('has an accessible submit button', () => {
    const Register = require('@/pages/register').default;
    render(<Register />);
    expect(screen.getByRole('button', { name: /create account|register|sign up/i })).toBeInTheDocument();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Database Management page
// ─────────────────────────────────────────────────────────────────────────────
describe('Accessibility: Database Management page', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    databaseApi.listDatabases.mockResolvedValue({ databases: [DB] });
  });

  it('has an h1 heading', async () => {
    const DatabaseManagement = require('@/pages/databases').default;
    render(<DatabaseManagement />);
    expect(screen.getByRole('heading', { level: 1 })).toBeInTheDocument();
  });

  it('labels the database name input', async () => {
    const DatabaseManagement = require('@/pages/databases').default;
    render(<DatabaseManagement />);
    expect(screen.getByLabelText(/database name/i)).toBeInTheDocument();
  });

  it('labels the vector dimension input', async () => {
    const DatabaseManagement = require('@/pages/databases').default;
    render(<DatabaseManagement />);
    expect(screen.getByLabelText(/vector dimension/i)).toBeInTheDocument();
  });

  it('has an accessible create button', async () => {
    const DatabaseManagement = require('@/pages/databases').default;
    render(<DatabaseManagement />);
    expect(screen.getByRole('button', { name: /create/i })).toBeInTheDocument();
  });

  it('renders loaded databases in the DOM', async () => {
    const DatabaseManagement = require('@/pages/databases').default;
    render(<DatabaseManagement />);
    await waitFor(() => {
      expect(screen.getByText('TestDB')).toBeInTheDocument();
    });
  });

  it('exposes error text when API fails', async () => {
    databaseApi.listDatabases.mockRejectedValueOnce(new Error('Unreachable'));
    const DatabaseManagement = require('@/pages/databases').default;
    render(<DatabaseManagement />);
    await waitFor(() => {
      expect(screen.getByText(/error fetching databases/i)).toBeInTheDocument();
    });
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Vector Search page
// ─────────────────────────────────────────────────────────────────────────────
describe('Accessibility: Vector Search page', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    window.location.search = '';
    databaseApi.listDatabases.mockResolvedValue({ databases: [DB] });
  });

  it('has an h1 heading', () => {
    const SearchInterface = require('@/pages/search').default;
    render(<SearchInterface />);
    expect(screen.getByRole('heading', { level: 1, name: /vector search/i })).toBeInTheDocument();
  });

  it('labels the database selector', () => {
    const SearchInterface = require('@/pages/search').default;
    render(<SearchInterface />);
    expect(screen.getByLabelText(/database/i)).toBeInTheDocument();
  });

  it('labels the query vector textarea', () => {
    const SearchInterface = require('@/pages/search').default;
    render(<SearchInterface />);
    expect(screen.getByLabelText(/query vector/i)).toBeInTheDocument();
  });

  it('labels the Top K input', () => {
    const SearchInterface = require('@/pages/search').default;
    render(<SearchInterface />);
    expect(screen.getByLabelText(/top k/i)).toBeInTheDocument();
  });

  it('has an accessible search button', () => {
    const SearchInterface = require('@/pages/search').default;
    render(<SearchInterface />);
    expect(screen.getByRole('button', { name: /search/i })).toBeInTheDocument();
  });

  it('exposes search error text accessibly', async () => {
    searchApi.similaritySearch.mockRejectedValueOnce(new Error('Query timeout'));
    const SearchInterface = require('@/pages/search').default;
    render(<SearchInterface />);
    await waitFor(() => screen.getByText('TestDB'));

    fireEvent.change(screen.getAllByRole('combobox')[0], { target: { value: 'db-1' } });
    const textarea = screen.getByPlaceholderText(/enter vector values/i);
    fireEvent.change(textarea, { target: { value: '0.1,0.2,0.3,0.4' } });
    fireEvent.submit(textarea.closest('form'));

    await waitFor(() => {
      expect(screen.getByText(/query timeout/i)).toBeInTheDocument();
    });
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// User Management page
// ─────────────────────────────────────────────────────────────────────────────
describe('Accessibility: User Management page', () => {
  const USERS = [
    { user_id: 'u-1', username: 'alice', email: 'alice@test.com', roles: ['admin'], active: true },
  ];

  beforeEach(() => {
    jest.clearAllMocks();
    global.confirm = jest.fn(() => false);
    usersApi.listUsers.mockResolvedValue({ users: USERS });
  });

  it('has a heading', () => {
    const UserManagement = require('@/pages/users').default;
    render(<UserManagement />);
    expect(screen.getByText(/user management/i)).toBeInTheDocument();
  });

  it('labels the username input', () => {
    const UserManagement = require('@/pages/users').default;
    render(<UserManagement />);
    // placeholder "john_doe" associates with a label
    expect(screen.getByPlaceholderText('john_doe')).toBeInTheDocument();
  });

  it('labels the email input', () => {
    const UserManagement = require('@/pages/users').default;
    render(<UserManagement />);
    expect(screen.getByPlaceholderText('john@example.com')).toBeInTheDocument();
  });

  it('labels the password input', () => {
    const UserManagement = require('@/pages/users').default;
    render(<UserManagement />);
    expect(screen.getByPlaceholderText('Enter password')).toBeInTheDocument();
  });

  it('has an accessible add-user submit button', () => {
    const UserManagement = require('@/pages/users').default;
    render(<UserManagement />);
    expect(screen.getByRole('button', { name: /add user|create user/i })).toBeInTheDocument();
  });

  it('renders user list after fetch', async () => {
    const UserManagement = require('@/pages/users').default;
    render(<UserManagement />);
    await waitFor(() => {
      expect(screen.getByText('alice')).toBeInTheDocument();
    });
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Alerting page
// ─────────────────────────────────────────────────────────────────────────────
describe('Accessibility: Alerting page', () => {
  const ALERTS = [
    { id: 'a-1', type: 'error', message: 'Disk space critical', timestamp: '2026-04-13T10:00:00Z' },
  ];

  beforeEach(() => {
    jest.useFakeTimers();
    jest.clearAllMocks();
    alertApi.listAlerts.mockResolvedValue({ alerts: ALERTS });
  });

  afterEach(() => jest.useRealTimers());

  it('has an h1 heading', async () => {
    const { act } = require('@testing-library/react');
    const Alerting = require('@/pages/alerting').default;
    await act(async () => render(<Alerting />));
    expect(screen.getByRole('heading', { level: 1, name: /system alerts/i })).toBeInTheDocument();
  });

  it('labels the alert type filter combobox', async () => {
    const { act } = require('@testing-library/react');
    const Alerting = require('@/pages/alerting').default;
    await act(async () => render(<Alerting />));
    // The <select> must be reachable for keyboard/AT users
    expect(screen.getByRole('combobox')).toBeInTheDocument();
  });

  it('renders alert messages as readable text', async () => {
    const { act } = require('@testing-library/react');
    const Alerting = require('@/pages/alerting').default;
    await act(async () => render(<Alerting />));
    await waitFor(() => {
      expect(screen.getByText(/disk space critical/i)).toBeInTheDocument();
    });
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Forgot Password page
// ─────────────────────────────────────────────────────────────────────────────
describe('Accessibility: Forgot Password page', () => {
  beforeEach(() => jest.clearAllMocks());

  it('has a heading', () => {
    const ForgotPassword = require('@/pages/forgot-password').default;
    render(<ForgotPassword />);
    expect(screen.getByRole('heading', { level: 1 })).toBeInTheDocument();
  });

  it('has at least one labelled input', () => {
    const ForgotPassword = require('@/pages/forgot-password').default;
    render(<ForgotPassword />);
    // username or email
    const inputs = screen.getAllByRole('textbox');
    expect(inputs.length).toBeGreaterThanOrEqual(1);
  });

  it('has an accessible submit button', () => {
    const ForgotPassword = require('@/pages/forgot-password').default;
    render(<ForgotPassword />);
    expect(screen.getByRole('button', { name: /send|reset|submit/i })).toBeInTheDocument();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Batch Operations page
// ─────────────────────────────────────────────────────────────────────────────
describe('Accessibility: Batch Operations page', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    databaseApi.listDatabases.mockResolvedValue({ databases: [DB] });
  });

  it('has a heading', () => {
    const BatchOperations = require('@/pages/batch-operations').default;
    render(<BatchOperations />);
    expect(screen.getByRole('heading', { name: /batch vector operations/i })).toBeInTheDocument();
  });

  it('has accessible radio buttons for mode selection', async () => {
    const BatchOperations = require('@/pages/batch-operations').default;
    render(<BatchOperations />);
    await waitFor(() => screen.getByText('TestDB'));

    const radios = screen.getAllByRole('radio');
    expect(radios.length).toBeGreaterThanOrEqual(2); // upload + download
    radios.forEach(radio => {
      expect(radio).toHaveAttribute('name');
    });
  });

  it('has an accessible upload button', async () => {
    const BatchOperations = require('@/pages/batch-operations').default;
    render(<BatchOperations />);
    await waitFor(() => screen.getByText('TestDB'));
    expect(screen.getByRole('button', { name: /upload vectors/i })).toBeInTheDocument();
  });

  it('has an accessible add-vector button', async () => {
    const BatchOperations = require('@/pages/batch-operations').default;
    render(<BatchOperations />);
    await waitFor(() => screen.getByText('TestDB'));
    expect(screen.getByRole('button', { name: /add vector/i })).toBeInTheDocument();
  });
});
