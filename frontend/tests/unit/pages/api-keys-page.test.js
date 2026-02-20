// frontend/tests/unit/pages/api-keys-page.test.js
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import ApiKeyManagement from '@/pages/api-keys';

// Mock the API functions
jest.mock('@/lib/api', () => ({
  apiKeysApi: {
    listApiKeys: jest.fn(),
    createApiKey: jest.fn(),
    revokeApiKey: jest.fn(),
  },
  authApi: {
    getCurrentUser: jest.fn(),
  }
}));

import { apiKeysApi, authApi } from '@/lib/api';

// Mock localStorage
Object.defineProperty(window, 'localStorage', {
  value: {
    getItem: jest.fn(() => 'test-auth-token'),
  },
  writable: true,
});

// Mock next/router
const mockPush = jest.fn();
jest.mock('next/router', () => ({
  useRouter: () => ({
    query: {},
    push: mockPush,
  })
}));

// Mock clipboard API
Object.assign(navigator, {
  clipboard: {
    writeText: jest.fn(() => Promise.resolve()),
  }
});

// Mock window.confirm
beforeAll(() => {
  jest.spyOn(window, 'confirm').mockImplementation(() => true);
});

describe('API Key Management Page', () => {
  const mockUser = {
    user_id: 'user-123',
    username: 'testuser'
  };

  const mockApiKeys = [
    {
      key_id: 'key-1',
      description: 'Production Key',
      is_active: true,
      created_at: '2026-01-01T00:00:00Z',
      expires_at: '2026-02-01T00:00:00Z',
      permissions: ['read', 'write']
    },
    {
      key_id: 'key-2',
      description: 'Test Key',
      is_active: false,
      created_at: '2025-12-01T00:00:00Z',
      expires_at: '2026-01-01T00:00:00Z',
      permissions: ['read']
    }
  ];

  beforeEach(() => {
    jest.clearAllMocks();

    // Mock authenticated user
    authApi.getCurrentUser.mockReturnValue(mockUser);

    // Mock successful API responses
    apiKeysApi.listApiKeys.mockResolvedValue({
      api_keys: mockApiKeys
    });

    apiKeysApi.createApiKey.mockResolvedValue({
      key_id: 'new-key-id',
      api_key: 'jvdb_new_generated_key_123456',
      description: 'New Key'
    });

    apiKeysApi.revokeApiKey.mockResolvedValue({});
  });

  describe('Authentication Required', () => {
    test('shows login prompt when not authenticated', async () => {
      authApi.getCurrentUser.mockReturnValue(null);

      render(<ApiKeyManagement />);

      expect(screen.getByText('Authentication Required')).toBeInTheDocument();
      expect(screen.getByText(/You must be logged in/)).toBeInTheDocument();
    });

    test('shows login button when not authenticated', async () => {
      authApi.getCurrentUser.mockReturnValue(null);

      render(<ApiKeyManagement />);

      expect(screen.getByRole('button', { name: /go to login/i })).toBeInTheDocument();
    });

    test('redirects to login when button clicked', async () => {
      authApi.getCurrentUser.mockReturnValue(null);

      render(<ApiKeyManagement />);

      fireEvent.click(screen.getByRole('button', { name: /go to login/i }));

      expect(mockPush).toHaveBeenCalledWith('/');
    });
  });

  describe('Authenticated View', () => {
    test('renders page title', async () => {
      render(<ApiKeyManagement />);

      expect(screen.getByText('API Key Management')).toBeInTheDocument();
    });

    test('renders page description', async () => {
      render(<ApiKeyManagement />);

      expect(screen.getByText(/Generate and manage your API keys/)).toBeInTheDocument();
    });

    test('shows current user info', async () => {
      render(<ApiKeyManagement />);

      expect(screen.getByText(/Logged in as:/)).toBeInTheDocument();
      expect(screen.getByText(/testuser/)).toBeInTheDocument();
    });

    test('renders create API key form', async () => {
      render(<ApiKeyManagement />);

      expect(screen.getByText('Create New API Key')).toBeInTheDocument();
      expect(screen.getByLabelText(/description/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/permissions/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/validity period/i)).toBeInTheDocument();
    });
  });

  describe('API Keys List', () => {
    test('fetches and displays API keys', async () => {
      render(<ApiKeyManagement />);

      await waitFor(() => {
        expect(screen.getByText('Production Key')).toBeInTheDocument();
        expect(screen.getByText('Test Key')).toBeInTheDocument();
      });

      expect(apiKeysApi.listApiKeys).toHaveBeenCalledWith('user-123');
    });

    test('displays key status badges', async () => {
      render(<ApiKeyManagement />);

      await waitFor(() => {
        expect(screen.getByText('Active')).toBeInTheDocument();
        expect(screen.getByText('Revoked')).toBeInTheDocument();
      });
    });

    test('displays key permissions', async () => {
      render(<ApiKeyManagement />);

      await waitFor(() => {
        expect(screen.getByText('read, write')).toBeInTheDocument();
      });
    });

    test('shows empty state when no API keys', async () => {
      apiKeysApi.listApiKeys.mockResolvedValue({ api_keys: [] });

      render(<ApiKeyManagement />);

      await waitFor(() => {
        expect(screen.getByText(/No API keys found/)).toBeInTheDocument();
      });
    });

    test('shows loading state', async () => {
      apiKeysApi.listApiKeys.mockImplementation(() => new Promise(() => {}));

      render(<ApiKeyManagement />);

      expect(screen.getByText(/Loading API keys/)).toBeInTheDocument();
    });

    test('shows revoke button for active keys', async () => {
      render(<ApiKeyManagement />);

      await waitFor(() => {
        expect(screen.getByText('Production Key')).toBeInTheDocument();
      });

      const revokeButtons = screen.getAllByRole('button', { name: /revoke/i });
      expect(revokeButtons.length).toBe(1); // Only active key has revoke button
    });
  });

  describe('Create API Key', () => {
    test('creates API key with form data', async () => {
      render(<ApiKeyManagement />);

      await waitFor(() => {
        expect(screen.getByText('Production Key')).toBeInTheDocument();
      });

      // Fill in form
      fireEvent.change(screen.getByLabelText(/description/i), { target: { value: 'New Production Key' } });
      fireEvent.change(screen.getByLabelText(/permissions/i), { target: { value: 'read, write, delete' } });
      fireEvent.change(screen.getByLabelText(/validity period/i), { target: { value: '60' } });

      // Submit
      fireEvent.click(screen.getByRole('button', { name: /create api key/i }));

      await waitFor(() => {
        expect(apiKeysApi.createApiKey).toHaveBeenCalledWith(
          'user-123',
          ['read', 'write', 'delete'],
          'New Production Key',
          60
        );
      });
    });

    test('shows generated key after creation', async () => {
      render(<ApiKeyManagement />);

      await waitFor(() => {
        expect(screen.getByText('Production Key')).toBeInTheDocument();
      });

      fireEvent.click(screen.getByRole('button', { name: /create api key/i }));

      await waitFor(() => {
        expect(screen.getByText('New API Key Created')).toBeInTheDocument();
        expect(screen.getByText('jvdb_new_generated_key_123456')).toBeInTheDocument();
      });
    });

    test('shows copy to clipboard button', async () => {
      render(<ApiKeyManagement />);

      await waitFor(() => {
        expect(screen.getByText('Production Key')).toBeInTheDocument();
      });

      fireEvent.click(screen.getByRole('button', { name: /create api key/i }));

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /copy to clipboard/i })).toBeInTheDocument();
      });
    });

    test('copies key to clipboard', async () => {
      render(<ApiKeyManagement />);

      await waitFor(() => {
        expect(screen.getByText('Production Key')).toBeInTheDocument();
      });

      fireEvent.click(screen.getByRole('button', { name: /create api key/i }));

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /copy to clipboard/i })).toBeInTheDocument();
      });

      fireEvent.click(screen.getByRole('button', { name: /copy to clipboard/i }));

      expect(navigator.clipboard.writeText).toHaveBeenCalledWith('jvdb_new_generated_key_123456');
    });

    test('shows success message after creation', async () => {
      render(<ApiKeyManagement />);

      await waitFor(() => {
        expect(screen.getByText('Production Key')).toBeInTheDocument();
      });

      fireEvent.click(screen.getByRole('button', { name: /create api key/i }));

      await waitFor(() => {
        expect(screen.getByText(/API key created successfully/)).toBeInTheDocument();
      });
    });

    test('handles create error', async () => {
      apiKeysApi.createApiKey.mockRejectedValue(new Error('Failed to create'));

      render(<ApiKeyManagement />);

      await waitFor(() => {
        expect(screen.getByText('Production Key')).toBeInTheDocument();
      });

      fireEvent.click(screen.getByRole('button', { name: /create api key/i }));

      await waitFor(() => {
        expect(screen.getByText(/Error creating API key/)).toBeInTheDocument();
      });
    });
  });

  describe('Revoke API Key', () => {
    test('revokes key when confirmed', async () => {
      render(<ApiKeyManagement />);

      await waitFor(() => {
        expect(screen.getByText('Production Key')).toBeInTheDocument();
      });

      const revokeButton = screen.getByRole('button', { name: /revoke/i });
      fireEvent.click(revokeButton);

      await waitFor(() => {
        expect(apiKeysApi.revokeApiKey).toHaveBeenCalledWith('key-1');
      });
    });

    test('does not revoke when cancelled', async () => {
      window.confirm.mockReturnValueOnce(false);

      render(<ApiKeyManagement />);

      await waitFor(() => {
        expect(screen.getByText('Production Key')).toBeInTheDocument();
      });

      const revokeButton = screen.getByRole('button', { name: /revoke/i });
      fireEvent.click(revokeButton);

      expect(apiKeysApi.revokeApiKey).not.toHaveBeenCalled();
    });

    test('shows success message after revoking', async () => {
      render(<ApiKeyManagement />);

      await waitFor(() => {
        expect(screen.getByText('Production Key')).toBeInTheDocument();
      });

      const revokeButton = screen.getByRole('button', { name: /revoke/i });
      fireEvent.click(revokeButton);

      await waitFor(() => {
        expect(screen.getByText(/API key revoked successfully/)).toBeInTheDocument();
      });
    });

    test('handles revoke error', async () => {
      apiKeysApi.revokeApiKey.mockRejectedValue(new Error('Failed to revoke'));

      render(<ApiKeyManagement />);

      await waitFor(() => {
        expect(screen.getByText('Production Key')).toBeInTheDocument();
      });

      const revokeButton = screen.getByRole('button', { name: /revoke/i });
      fireEvent.click(revokeButton);

      await waitFor(() => {
        expect(screen.getByText(/Error revoking API key/)).toBeInTheDocument();
      });
    });
  });

  describe('Error Handling', () => {
    test('handles fetch API keys error', async () => {
      apiKeysApi.listApiKeys.mockRejectedValue(new Error('Network error'));

      render(<ApiKeyManagement />);

      await waitFor(() => {
        expect(screen.getByText(/Error fetching API keys/)).toBeInTheDocument();
      });
    });
  });

  describe('Date Formatting', () => {
    test('displays API keys with Unix timestamp 0 as Never for last_used_at', async () => {
      apiKeysApi.listApiKeys.mockResolvedValue({
        api_keys: [
          {
            key_id: 'key-ts',
            description: 'Timestamp Key',
            is_active: true,
            created_at: 1704067200, // Unix timestamp in seconds
            expires_at: '2026-06-01T00:00:00Z',
            last_used_at: 0, // Should show 'Never'
            permissions: ['read']
          }
        ]
      });

      render(<ApiKeyManagement />);

      await waitFor(() => {
        expect(screen.getByText('Timestamp Key')).toBeInTheDocument();
        expect(screen.getByText('Never')).toBeInTheDocument();
      });
    });

    test('displays API keys with ISO string timestamps', async () => {
      apiKeysApi.listApiKeys.mockResolvedValue({
        api_keys: [
          {
            key_id: 'key-iso',
            description: 'ISO Key',
            is_active: true,
            created_at: '2026-01-01T00:00:00Z',
            expires_at: '2026-06-01T00:00:00Z',
            last_used_at: '',
            permissions: ['read']
          }
        ]
      });

      render(<ApiKeyManagement />);

      await waitFor(() => {
        expect(screen.getByText('ISO Key')).toBeInTheDocument();
      });
    });
  });

  describe('Form Validation', () => {
    test('default validity days is 30', async () => {
      render(<ApiKeyManagement />);

      const validityInput = screen.getByLabelText(/validity period/i);
      expect(validityInput).toHaveValue(30);
    });

    test('clears form after successful creation', async () => {
      render(<ApiKeyManagement />);

      await waitFor(() => {
        expect(screen.getByText('Production Key')).toBeInTheDocument();
      });

      // Fill in form
      const descInput = screen.getByLabelText(/description/i);
      fireEvent.change(descInput, { target: { value: 'Test Description' } });

      // Submit
      fireEvent.click(screen.getByRole('button', { name: /create api key/i }));

      await waitFor(() => {
        expect(screen.getByText('New API Key Created')).toBeInTheDocument();
      });

      // Form should be cleared
      expect(descInput).toHaveValue('');
    });
  });
});
