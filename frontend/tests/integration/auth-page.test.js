// frontend/tests/integration/auth-page.test.js
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import AuthManagement from '@/pages/auth';

// Mock the API functions
jest.mock('@/lib/api', () => ({
  authApi: {
    login: jest.fn(),
    register: jest.fn(),
    logout: jest.fn(),
  },
  apiKeyApi: {
    listKeys: jest.fn(),
    createKey: jest.fn(),
    deleteKey: jest.fn(),
  }
}));

import { authApi, apiKeyApi } from '@/lib/api';

// Mock localStorage for authentication tests
const mockLocalStorage = (() => {
  let store = {};

  return {
    getItem: jest.fn((key) => store[key] || null),
    setItem: jest.fn((key, value) => {
      store[key] = value.toString();
    }),
    removeItem: jest.fn((key) => {
      delete store[key];
    }),
    clear: jest.fn(() => {
      store = {};
    }),
  };
})();

Object.defineProperty(window, 'localStorage', {
  value: mockLocalStorage,
});

// Mock clipboard API
Object.assign(navigator, {
  clipboard: {
    writeText: jest.fn(() => Promise.resolve()),
  }
});

// Mock next/router
jest.mock('next/router', () => ({
  useRouter: () => ({
    query: {},
    push: jest.fn(),
  })
}));

// Mock window.alert
beforeAll(() => {
  jest.spyOn(window, 'alert').mockImplementation(() => {});
});

describe('Auth Management Page Integration', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockLocalStorage.clear();

    // Mock successful API responses
    apiKeyApi.listKeys.mockResolvedValue({ apiKeys: [] });
    authApi.login.mockResolvedValue({ token: 'test-token' });
    apiKeyApi.createKey.mockResolvedValue({
      keyId: 'new-key-id',
      name: 'Test Key',
      apiKey: 'jvdb_generated_key_123'
    });
  });

  test('renders auth page', () => {
    render(<AuthManagement />);

    // Check that the page renders without crashing
    expect(screen.getByRole('button', { name: /authentication/i })).toBeInTheDocument();
  });

  test('allows user to switch to authentication tab', () => {
    render(<AuthManagement />);

    // Click on Authentication tab button
    fireEvent.click(screen.getByRole('button', { name: /authentication/i }));

    // Check that login button is visible
    expect(screen.getByRole('button', { name: /log in/i })).toBeInTheDocument();
  });

  test('renders registration form fields in auth tab', () => {
    render(<AuthManagement />);

    // Switch to Authentication tab
    fireEvent.click(screen.getByRole('button', { name: /authentication/i }));

    // Check for registration form elements
    expect(screen.getByLabelText(/confirm password/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /register/i })).toBeInTheDocument();
  });

  test('calls login API when login form is submitted', async () => {
    render(<AuthManagement />);

    // Switch to Authentication tab
    fireEvent.click(screen.getByRole('button', { name: /authentication/i }));

    // Fill in login form - use getAllByLabelText if there are multiple
    const usernameInputs = screen.getAllByLabelText(/username/i);
    const passwordInputs = screen.getAllByLabelText(/^password$/i);

    // Use the first matching input (login form)
    fireEvent.change(usernameInputs[0], { target: { value: 'testuser' } });
    fireEvent.change(passwordInputs[0], { target: { value: 'password123' } });

    // Submit login
    fireEvent.click(screen.getByRole('button', { name: /log in/i }));

    // Wait for API call
    await waitFor(() => {
      expect(authApi.login).toHaveBeenCalledWith('testuser', 'password123');
    });
  });

  test('renders API key tab button', () => {
    render(<AuthManagement />);

    // API Keys tab button should be present
    expect(screen.getByRole('button', { name: /api keys/i })).toBeInTheDocument();
  });
});
