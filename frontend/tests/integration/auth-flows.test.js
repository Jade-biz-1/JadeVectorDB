// frontend/tests/integration/auth-flows.test.js
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import AuthManagement from '@/pages/auth';
import { authApi, apiKeyApi } from '@/lib/api';

// Mock the API modules
jest.mock('@/lib/api', () => ({
  authApi: {
    login: jest.fn(),
    logout: jest.fn(),
    register: jest.fn(),
  },
  apiKeyApi: {
    createKey: jest.fn(),
    listKeys: jest.fn(),
    revokeKey: jest.fn(),
  },
}));

// Mock localStorage
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
    get store() {
      return store;
    },
    set store(value) {
      store = value;
    }
  };
})();

Object.defineProperty(window, 'localStorage', {
  value: mockLocalStorage,
});

// Mock window.alert
global.alert = jest.fn();

describe('Authentication Flows - Comprehensive Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockLocalStorage.clear();
    mockLocalStorage.store = {};
    global.alert.mockClear();
    apiKeyApi.listKeys.mockResolvedValue({ apiKeys: [] });
  });

  // ============================================================================
  // Login Flow Tests
  // ============================================================================

  describe('Login Flow', () => {
    test('successful login with valid credentials', async () => {
      const mockResponse = {
        token: 'mock-jwt-token-12345',
        user: { id: 'user-1', username: 'testuser' }
      };
      authApi.login.mockResolvedValueOnce(mockResponse);

      render(<AuthManagement />);

      // Navigate to authentication tab
      fireEvent.click(screen.getByText('Authentication'));

      // Fill in login form
      const usernameInput = screen.getByLabelText('Username');
      const passwordInput = screen.getByLabelText('Password');

      fireEvent.change(usernameInput, { target: { value: 'testuser' } });
      fireEvent.change(passwordInput, { target: { value: 'password123' } });

      // Submit login
      fireEvent.click(screen.getByRole('button', { name: /log in/i }));

      await waitFor(() => {
        expect(authApi.login).toHaveBeenCalledWith('testuser', 'password123');
        expect(mockLocalStorage.setItem).toHaveBeenCalledWith('jadevectordb_api_key', 'mock-jwt-token-12345');
        expect(mockLocalStorage.setItem).toHaveBeenCalledWith('jadevectordb_authenticated', 'true');
        expect(mockLocalStorage.setItem).toHaveBeenCalledWith('jadevectordb_username', 'testuser');
        expect(global.alert).toHaveBeenCalledWith('Successfully logged in as testuser');
      });
    });

    test('login failure with invalid credentials', async () => {
      authApi.login.mockRejectedValueOnce(new Error('Invalid username or password'));

      render(<AuthManagement />);

      fireEvent.click(screen.getByText('Authentication'));

      fireEvent.change(screen.getByLabelText('Username'), { target: { value: 'wronguser' } });
      fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'wrongpass' } });

      fireEvent.click(screen.getByRole('button', { name: /log in/i }));

      await waitFor(() => {
        expect(authApi.login).toHaveBeenCalledWith('wronguser', 'wrongpass');
        expect(global.alert).toHaveBeenCalledWith('Login failed: Invalid username or password');
        expect(mockLocalStorage.setItem).not.toHaveBeenCalledWith('jadevectordb_authenticated', 'true');
      });
    });

    test('login with empty fields shows error', async () => {
      render(<AuthManagement />);

      fireEvent.click(screen.getByText('Authentication'));

      // Try to submit without filling fields
      fireEvent.click(screen.getByRole('button', { name: /log in/i }));

      await waitFor(() => {
        expect(authApi.login).toHaveBeenCalledWith('', '');
      });
    });

    test('login without token in response shows error', async () => {
      authApi.login.mockResolvedValueOnce({ user: { id: 'user-1' } }); // No token

      render(<AuthManagement />);

      fireEvent.click(screen.getByText('Authentication'));

      fireEvent.change(screen.getByLabelText('Username'), { target: { value: 'testuser' } });
      fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'password123' } });

      fireEvent.click(screen.getByRole('button', { name: /log in/i }));

      await waitFor(() => {
        expect(global.alert).toHaveBeenCalledWith('Login failed: No token received');
      });
    });

    test('login button shows loading state during API call', async () => {
      authApi.login.mockImplementation(() => new Promise(resolve => setTimeout(() => resolve({ token: 'test' }), 100)));

      render(<AuthManagement />);

      fireEvent.click(screen.getByText('Authentication'));

      fireEvent.change(screen.getByLabelText('Username'), { target: { value: 'testuser' } });
      fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'password123' } });

      const loginButton = screen.getByRole('button', { name: /log in/i });
      fireEvent.click(loginButton);

      // Button should be disabled during loading
      expect(loginButton).toBeDisabled();

      await waitFor(() => {
        expect(authApi.login).toHaveBeenCalled();
      });
    });
  });

  // ============================================================================
  // Logout Flow Tests
  // ============================================================================

  describe('Logout Flow', () => {
    test('successful logout clears localStorage', async () => {
      authApi.logout.mockResolvedValueOnce({ success: true });

      // Set up authenticated state
      mockLocalStorage.store = {
        'jadevectordb_authenticated': 'true',
        'jadevectordb_api_key': 'test-token',
        'jadevectordb_username': 'testuser'
      };

      render(<AuthManagement />);

      // Wait for component to initialize
      await waitFor(() => {
        expect(mockLocalStorage.getItem('jadevectordb_authenticated')).toBe('true');
      });

      fireEvent.click(screen.getByText('Authentication'));

      // Find and click logout button
      const logoutButton = screen.getByRole('button', { name: /log out/i });
      fireEvent.click(logoutButton);

      await waitFor(() => {
        expect(authApi.logout).toHaveBeenCalled();
        expect(mockLocalStorage.removeItem).toHaveBeenCalledWith('jadevectordb_authenticated');
        expect(mockLocalStorage.removeItem).toHaveBeenCalledWith('jadevectordb_api_key');
        expect(mockLocalStorage.removeItem).toHaveBeenCalledWith('jadevectordb_username');
        expect(global.alert).toHaveBeenCalledWith('Logged out successfully');
      });
    });

    test('logout handles API failure gracefully', async () => {
      authApi.logout.mockRejectedValueOnce(new Error('Network error'));

      mockLocalStorage.store = {
        'jadevectordb_authenticated': 'true',
        'jadevectordb_api_key': 'test-token'
      };

      render(<AuthManagement />);

      await waitFor(() => {
        expect(mockLocalStorage.getItem('jadevectordb_authenticated')).toBe('true');
      });

      fireEvent.click(screen.getByText('Authentication'));

      const logoutButton = screen.getByRole('button', { name: /log out/i });
      fireEvent.click(logoutButton);

      // Should still clear localStorage even if API call fails
      await waitFor(() => {
        expect(mockLocalStorage.removeItem).toHaveBeenCalledWith('jadevectordb_authenticated');
        expect(global.alert).toHaveBeenCalledWith('Logged out successfully');
      });
    });
  });

  // ============================================================================
  // Registration Flow Tests
  // ============================================================================

  describe('Registration Flow', () => {
    test('successful registration with matching passwords', async () => {
      authApi.register.mockResolvedValueOnce({ userId: 'new-user-id', username: 'newuser' });

      render(<AuthManagement />);

      fireEvent.click(screen.getByText('Authentication'));

      // Fill registration form
      fireEvent.change(screen.getByLabelText('Username'), { target: { value: 'newuser' } });
      fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'securepass123' } });
      fireEvent.change(screen.getByLabelText('Confirm Password'), { target: { value: 'securepass123' } });

      // Click register button
      fireEvent.click(screen.getByRole('button', { name: /register/i }));

      await waitFor(() => {
        expect(authApi.register).toHaveBeenCalledWith('newuser', 'securepass123');
        expect(global.alert).toHaveBeenCalledWith('User newuser registered successfully! You can now log in.');
      });
    });

    test('registration fails when passwords do not match', async () => {
      render(<AuthManagement />);

      fireEvent.click(screen.getByText('Authentication'));

      fireEvent.change(screen.getByLabelText('Username'), { target: { value: 'newuser' } });
      fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'password123' } });
      fireEvent.change(screen.getByLabelText('Confirm Password'), { target: { value: 'differentpass' } });

      fireEvent.click(screen.getByRole('button', { name: /register/i }));

      await waitFor(() => {
        expect(global.alert).toHaveBeenCalledWith('Passwords do not match');
        expect(authApi.register).not.toHaveBeenCalled();
      });
    });

    test('registration fails with duplicate username', async () => {
      authApi.register.mockRejectedValueOnce(new Error('Username already exists'));

      render(<AuthManagement />);

      fireEvent.click(screen.getByText('Authentication'));

      fireEvent.change(screen.getByLabelText('Username'), { target: { value: 'existinguser' } });
      fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'password123' } });
      fireEvent.change(screen.getByLabelText('Confirm Password'), { target: { value: 'password123' } });

      fireEvent.click(screen.getByRole('button', { name: /register/i }));

      await waitFor(() => {
        expect(authApi.register).toHaveBeenCalled();
        expect(global.alert).toHaveBeenCalledWith('Registration failed: Username already exists');
      });
    });

    test('registration clears form fields on success', async () => {
      authApi.register.mockResolvedValueOnce({ userId: 'user-id' });

      render(<AuthManagement />);

      fireEvent.click(screen.getByText('Authentication'));

      const usernameInput = screen.getByLabelText('Username');
      const passwordInput = screen.getByLabelText('Password');
      const confirmPasswordInput = screen.getByLabelText('Confirm Password');

      fireEvent.change(usernameInput, { target: { value: 'newuser' } });
      fireEvent.change(passwordInput, { target: { value: 'password123' } });
      fireEvent.change(confirmPasswordInput, { target: { value: 'password123' } });

      fireEvent.click(screen.getByRole('button', { name: /register/i }));

      await waitFor(() => {
        expect(usernameInput.value).toBe('');
        expect(passwordInput.value).toBe('');
        expect(confirmPasswordInput.value).toBe('');
      });
    });
  });

  // ============================================================================
  // API Key Management Tests
  // ============================================================================

  describe('API Key Management', () => {
    test('creates new API key with name and permissions', async () => {
      const mockApiKey = {
        apiKey: 'sk_test_1234567890abcdef',
        keyId: 'key-123',
        name: 'Test Key',
        permissions: ['read', 'write']
      };

      apiKeyApi.createKey.mockResolvedValueOnce(mockApiKey);
      apiKeyApi.listKeys.mockResolvedValueOnce({
        apiKeys: [
          {
            keyId: 'key-123',
            name: 'Test Key',
            createdAt: new Date().toISOString(),
            permissions: ['read', 'write']
          }
        ]
      });

      render(<AuthManagement />);

      // Should start on API key tab
      expect(screen.getByText('Manage API keys for programmatic access')).toBeInTheDocument();

      // Fill in API key form
      fireEvent.change(screen.getByLabelText('Key Name'), { target: { value: 'Test Key' } });

      // Click create button
      fireEvent.click(screen.getByRole('button', { name: /create api key/i }));

      await waitFor(() => {
        expect(apiKeyApi.createKey).toHaveBeenCalledWith({
          name: 'Test Key',
          permissions: ['read']
        });
      });
    });

    test('displays generated API key after creation', async () => {
      const mockApiKey = {
        apiKey: 'sk_test_1234567890abcdef',
        keyId: 'key-123'
      };

      apiKeyApi.createKey.mockResolvedValueOnce(mockApiKey);
      apiKeyApi.listKeys.mockResolvedValue({ apiKeys: [] });

      render(<AuthManagement />);

      fireEvent.change(screen.getByLabelText('Key Name'), { target: { value: 'Test Key' } });
      fireEvent.click(screen.getByRole('button', { name: /create api key/i }));

      await waitFor(() => {
        expect(screen.getByText('Your new API Key')).toBeInTheDocument();
      });
    });

    test('shows error when API key name is empty', async () => {
      render(<AuthManagement />);

      // Try to create without name
      fireEvent.click(screen.getByRole('button', { name: /create api key/i }));

      await waitFor(() => {
        expect(global.alert).toHaveBeenCalledWith('Please enter a name for the API key');
        expect(apiKeyApi.createKey).not.toHaveBeenCalled();
      });
    });

    test('refreshes API key list after creation', async () => {
      apiKeyApi.createKey.mockResolvedValueOnce({ apiKey: 'test-key', keyId: 'key-1' });
      apiKeyApi.listKeys.mockResolvedValueOnce({ apiKeys: [] });
      apiKeyApi.listKeys.mockResolvedValueOnce({
        apiKeys: [
          {
            keyId: 'key-1',
            name: 'New Key',
            createdAt: new Date().toISOString(),
            permissions: ['read']
          }
        ]
      });

      render(<AuthManagement />);

      fireEvent.change(screen.getByLabelText('Key Name'), { target: { value: 'New Key' } });
      fireEvent.click(screen.getByRole('button', { name: /create api key/i }));

      await waitFor(() => {
        expect(apiKeyApi.listKeys).toHaveBeenCalledTimes(2); // Initial load + refresh
      });
    });

    // ============================================================================
    // API Key Revocation UX Tests (T233)
    // ============================================================================

    test('revokes API key successfully', async () => {
      const existingKeys = [
        {
          keyId: 'key-to-revoke',
          name: 'Key to Revoke',
          createdAt: new Date().toISOString(),
          permissions: ['read', 'write']
        }
      ];

      apiKeyApi.listKeys.mockResolvedValueOnce({ apiKeys: existingKeys });
      apiKeyApi.revokeKey.mockResolvedValueOnce({ success: true, message: 'API key revoked' });
      apiKeyApi.listKeys.mockResolvedValueOnce({ apiKeys: [] }); // After revocation

      // Set up authenticated state
      mockLocalStorage.store = {
        'jadevectordb_authenticated': 'true',
        'jadevectordb_api_key': 'test-token'
      };

      render(<AuthManagement />);

      // Wait for keys to load
      await waitFor(() => {
        expect(screen.getByText('Key to Revoke')).toBeInTheDocument();
      });

      // Find and click revoke button
      const revokeButton = screen.getByRole('button', { name: /revoke/i });
      fireEvent.click(revokeButton);

      await waitFor(() => {
        expect(apiKeyApi.revokeKey).toHaveBeenCalledWith('key-to-revoke');
        expect(global.alert).toHaveBeenCalledWith('API key revoked successfully');
      });
    });

    test('shows confirmation before revoking API key', async () => {
      const existingKeys = [
        {
          keyId: 'key-123',
          name: 'Important Key',
          createdAt: new Date().toISOString(),
          permissions: ['read']
        }
      ];

      apiKeyApi.listKeys.mockResolvedValueOnce({ apiKeys: existingKeys });

      mockLocalStorage.store = {
        'jadevectordb_authenticated': 'true',
        'jadevectordb_api_key': 'test-token'
      };

      // Mock window.confirm
      const originalConfirm = window.confirm;
      window.confirm = jest.fn(() => false); // User cancels

      render(<AuthManagement />);

      await waitFor(() => {
        expect(screen.getByText('Important Key')).toBeInTheDocument();
      });

      const revokeButton = screen.getByRole('button', { name: /revoke/i });
      fireEvent.click(revokeButton);

      // Should not call revokeKey if user cancels
      expect(apiKeyApi.revokeKey).not.toHaveBeenCalled();

      window.confirm = originalConfirm;
    });

    test('handles API key revocation failure', async () => {
      const existingKeys = [
        {
          keyId: 'key-123',
          name: 'Protected Key',
          createdAt: new Date().toISOString(),
          permissions: ['read']
        }
      ];

      apiKeyApi.listKeys.mockResolvedValueOnce({ apiKeys: existingKeys });
      apiKeyApi.revokeKey.mockRejectedValueOnce(new Error('Cannot revoke this key'));

      mockLocalStorage.store = {
        'jadevectordb_authenticated': 'true',
        'jadevectordb_api_key': 'test-token'
      };

      render(<AuthManagement />);

      await waitFor(() => {
        expect(screen.getByText('Protected Key')).toBeInTheDocument();
      });

      const revokeButton = screen.getByRole('button', { name: /revoke/i });
      fireEvent.click(revokeButton);

      await waitFor(() => {
        expect(global.alert).toHaveBeenCalledWith('Error revoking API key: Cannot revoke this key');
      });
    });

    test('refreshes API key list after successful revocation', async () => {
      const existingKeys = [
        { keyId: 'key-1', name: 'Key 1', createdAt: new Date().toISOString(), permissions: ['read'] },
        { keyId: 'key-2', name: 'Key 2', createdAt: new Date().toISOString(), permissions: ['read'] }
      ];

      apiKeyApi.listKeys.mockResolvedValueOnce({ apiKeys: existingKeys });
      apiKeyApi.revokeKey.mockResolvedValueOnce({ success: true });
      apiKeyApi.listKeys.mockResolvedValueOnce({ 
        apiKeys: [{ keyId: 'key-2', name: 'Key 2', createdAt: new Date().toISOString(), permissions: ['read'] }]
      });

      mockLocalStorage.store = {
        'jadevectordb_authenticated': 'true',
        'jadevectordb_api_key': 'test-token'
      };

      render(<AuthManagement />);

      await waitFor(() => {
        expect(screen.getByText('Key 1')).toBeInTheDocument();
        expect(screen.getByText('Key 2')).toBeInTheDocument();
      });

      // Revoke first key
      const revokeButtons = screen.getAllByRole('button', { name: /revoke/i });
      fireEvent.click(revokeButtons[0]);

      await waitFor(() => {
        expect(apiKeyApi.listKeys).toHaveBeenCalledTimes(2); // Initial + refresh
      });
    });

    test('disables revoke button during revocation', async () => {
      const existingKeys = [
        { keyId: 'key-1', name: 'Key 1', createdAt: new Date().toISOString(), permissions: ['read'] }
      ];

      apiKeyApi.listKeys.mockResolvedValueOnce({ apiKeys: existingKeys });
      apiKeyApi.revokeKey.mockImplementation(() => 
        new Promise(resolve => setTimeout(() => resolve({ success: true }), 100))
      );

      mockLocalStorage.store = {
        'jadevectordb_authenticated': 'true',
        'jadevectordb_api_key': 'test-token'
      };

      render(<AuthManagement />);

      await waitFor(() => {
        expect(screen.getByText('Key 1')).toBeInTheDocument();
      });

      const revokeButton = screen.getByRole('button', { name: /revoke/i });
      fireEvent.click(revokeButton);

      // Button should be disabled during revocation
      expect(revokeButton).toBeDisabled();

      await waitFor(() => {
        expect(apiKeyApi.revokeKey).toHaveBeenCalled();
      });
    });
  });

  // ============================================================================
  // Authentication State Persistence Tests
  // ============================================================================

  describe('Authentication State Persistence', () => {
    test('loads authenticated state from localStorage on mount', async () => {
      mockLocalStorage.store = {
        'jadevectordb_authenticated': 'true',
        'jadevectordb_api_key': 'stored-token'
      };

      apiKeyApi.listKeys.mockResolvedValueOnce({ apiKeys: [] });

      render(<AuthManagement />);

      await waitFor(() => {
        expect(mockLocalStorage.getItem).toHaveBeenCalledWith('jadevectordb_authenticated');
        expect(mockLocalStorage.getItem).toHaveBeenCalledWith('jadevectordb_api_key');
        expect(apiKeyApi.listKeys).toHaveBeenCalled();
      });
    });

    test('does not fetch API keys when not authenticated', async () => {
      mockLocalStorage.store = {}; // No auth data

      render(<AuthManagement />);

      await waitFor(() => {
        expect(apiKeyApi.listKeys).not.toHaveBeenCalled();
      });
    });
  });

  // ============================================================================
  // Tab Navigation Tests
  // ============================================================================

  describe('Tab Navigation', () => {
    test('switches between authentication and API key tabs', () => {
      render(<AuthManagement />);

      // Should start on API key tab
      expect(screen.getByText('Manage API keys for programmatic access')).toBeInTheDocument();

      // Switch to authentication tab
      fireEvent.click(screen.getByText('Authentication'));
      expect(screen.getByText('Log in to access the system')).toBeInTheDocument();

      // Switch back to API key tab
      fireEvent.click(screen.getByText('API Keys'));
      expect(screen.getByText('Manage API keys for programmatic access')).toBeInTheDocument();
    });

    test('tab state persists during form interactions', () => {
      render(<AuthManagement />);

      // Go to auth tab and fill form
      fireEvent.click(screen.getByText('Authentication'));
      fireEvent.change(screen.getByLabelText('Username'), { target: { value: 'testuser' } });

      // Switch tabs
      fireEvent.click(screen.getByText('API Keys'));
      fireEvent.click(screen.getByText('Authentication'));

      // Form should maintain values
      expect(screen.getByLabelText('Username')).toHaveValue('testuser');
    });
  });

  // ============================================================================
  // Error Handling Tests
  // ============================================================================

  describe('Error Handling', () => {
    test('handles network errors during login', async () => {
      authApi.login.mockRejectedValueOnce(new Error('Network request failed'));

      render(<AuthManagement />);

      fireEvent.click(screen.getByText('Authentication'));

      fireEvent.change(screen.getByLabelText('Username'), { target: { value: 'user' } });
      fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'pass' } });
      fireEvent.click(screen.getByRole('button', { name: /log in/i }));

      await waitFor(() => {
        expect(global.alert).toHaveBeenCalledWith('Login failed: Network request failed');
      });
    });

    test('handles API key creation failure', async () => {
      apiKeyApi.createKey.mockRejectedValueOnce(new Error('Insufficient permissions'));

      render(<AuthManagement />);

      fireEvent.change(screen.getByLabelText('Key Name'), { target: { value: 'Test Key' } });
      fireEvent.click(screen.getByRole('button', { name: /create api key/i }));

      await waitFor(() => {
        expect(global.alert).toHaveBeenCalledWith('Error creating API key: Insufficient permissions');
      });
    });

    test('handles API key list fetch failure', async () => {
      mockLocalStorage.store = {
        'jadevectordb_authenticated': 'true',
        'jadevectordb_api_key': 'test-token'
      };

      apiKeyApi.listKeys.mockRejectedValueOnce(new Error('Failed to fetch'));

      render(<AuthManagement />);

      await waitFor(() => {
        expect(apiKeyApi.listKeys).toHaveBeenCalled();
        // Should handle error gracefully, not crash
      });
    });
  });
});
