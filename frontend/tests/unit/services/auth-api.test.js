// frontend/tests/unit/services/auth-api.test.js
import { authApi, apiKeyApi } from '@/lib/api';

// Mock fetch globally
global.fetch = jest.fn();

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
  writable: true
});

describe('Auth API Service', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockLocalStorage.clear();
    mockLocalStorage.store = {};
    global.fetch.mockClear();
  });

  // ============================================================================
  // Authentication API Tests
  // ============================================================================

  describe('authApi.login', () => {
    test('sends POST request with credentials', async () => {
      const mockResponse = {
        token: 'jwt-token-123',
        user: { id: 'user-1', username: 'testuser' }
      };

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse
      });

      const result = await authApi.login('testuser', 'password123');

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/v1/auth/login'),
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json'
          }),
          body: JSON.stringify({
            username: 'testuser',
            password: 'password123'
          })
        })
      );

      expect(result).toEqual(mockResponse);
    });

    test('throws error on failed login', async () => {
      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: async () => ({ message: 'Invalid credentials' })
      });

      await expect(authApi.login('wronguser', 'wrongpass'))
        .rejects.toThrow('Invalid credentials');
    });

    test('handles network errors', async () => {
      global.fetch.mockRejectedValueOnce(new Error('Network error'));

      await expect(authApi.login('user', 'pass'))
        .rejects.toThrow('Network error');
    });

    test('handles empty response body', async () => {
      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: async () => { throw new Error('No JSON'); }
      });

      await expect(authApi.login('user', 'pass'))
        .rejects.toThrow();
    });
  });

  describe('authApi.register', () => {
    test('sends POST request with registration data', async () => {
      const mockResponse = {
        userId: 'new-user-id',
        username: 'newuser',
        message: 'Registration successful'
      };

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse
      });

      const result = await authApi.register('newuser', 'securepass123', ['user']);

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/v1/auth/register'),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            username: 'newuser',
            password: 'securepass123',
            roles: ['user']
          })
        })
      );

      expect(result).toEqual(mockResponse);
    });

    test('uses default role when roles not provided', async () => {
      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ userId: 'id' })
      });

      await authApi.register('user', 'pass');

      const callArgs = global.fetch.mock.calls[0][1].body;
      const parsedBody = JSON.parse(callArgs);
      expect(parsedBody.roles).toEqual(['user']);
    });

    test('handles duplicate username error', async () => {
      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 409,
        json: async () => ({ message: 'Username already exists' })
      });

      await expect(authApi.register('existinguser', 'pass'))
        .rejects.toThrow('Username already exists');
    });

    test('handles weak password error', async () => {
      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => ({ message: 'Password does not meet requirements' })
      });

      await expect(authApi.register('user', 'weak'))
        .rejects.toThrow('Password does not meet requirements');
    });
  });

  describe('authApi.logout', () => {
    test('sends POST request with auth token', async () => {
      mockLocalStorage.store = { 'jadevectordb_api_key': 'test-token' };

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true })
      });

      const result = await authApi.logout();

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/v1/auth/logout'),
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'X-API-Key': 'test-token'
          })
        })
      );

      expect(result).toEqual({ success: true });
    });

    test('works without API key', async () => {
      mockLocalStorage.store = {}; // No API key

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true })
      });

      await authApi.logout();

      expect(global.fetch).toHaveBeenCalled();
    });

    test('handles logout failure', async () => {
      mockLocalStorage.store = { 'jadevectordb_api_key': 'test-token' };

      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: async () => ({ message: 'Invalid token' })
      });

      await expect(authApi.logout()).rejects.toThrow('Invalid token');
    });
  });

  // ============================================================================
  // API Key Management Tests
  // ============================================================================

  describe('apiKeyApi.createKey', () => {
    test('creates API key with name and permissions', async () => {
      mockLocalStorage.store = { 'jadevectordb_api_key': 'admin-token' };

      const mockResponse = {
        apiKey: 'sk_test_1234567890',
        keyId: 'key-123',
        name: 'Production Key',
        permissions: ['read', 'write']
      };

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse
      });

      const keyData = {
        name: 'Production Key',
        permissions: ['read', 'write'],
        description: 'Key for production use'
      };

      const result = await apiKeyApi.createKey(keyData);

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/v1/apikeys'),
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'X-API-Key': 'admin-token'
          }),
          body: JSON.stringify(keyData)
        })
      );

      expect(result).toEqual(mockResponse);
    });

    test('requires authentication', async () => {
      mockLocalStorage.store = {}; // No API key

      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: async () => ({ message: 'Authentication required' })
      });

      await expect(apiKeyApi.createKey({ name: 'Test' }))
        .rejects.toThrow('Authentication required');
    });

    test('handles permission denied', async () => {
      mockLocalStorage.store = { 'jadevectordb_api_key': 'limited-token' };

      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 403,
        json: async () => ({ message: 'Insufficient permissions' })
      });

      await expect(apiKeyApi.createKey({ name: 'Test' }))
        .rejects.toThrow('Insufficient permissions');
    });
  });

  describe('apiKeyApi.listKeys', () => {
    test('fetches list of API keys', async () => {
      mockLocalStorage.store = { 'jadevectordb_api_key': 'test-token' };

      const mockResponse = {
        apiKeys: [
          {
            keyId: 'key-1',
            name: 'Key 1',
            createdAt: '2025-01-01T00:00:00Z',
            permissions: ['read']
          },
          {
            keyId: 'key-2',
            name: 'Key 2',
            createdAt: '2025-01-02T00:00:00Z',
            permissions: ['read', 'write']
          }
        ],
        total: 2
      };

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse
      });

      const result = await apiKeyApi.listKeys();

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/v1/apikeys'),
        expect.objectContaining({
          method: 'GET',
          headers: expect.objectContaining({
            'X-API-Key': 'test-token'
          })
        })
      );

      expect(result).toEqual(mockResponse);
      expect(result.apiKeys).toHaveLength(2);
    });

    test('returns empty array when no keys exist', async () => {
      mockLocalStorage.store = { 'jadevectordb_api_key': 'test-token' };

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ apiKeys: [], total: 0 })
      });

      const result = await apiKeyApi.listKeys();

      expect(result.apiKeys).toEqual([]);
      expect(result.total).toBe(0);
    });

    test('handles authentication failure', async () => {
      mockLocalStorage.store = { 'jadevectordb_api_key': 'expired-token' };

      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: async () => ({ message: 'Token expired' })
      });

      await expect(apiKeyApi.listKeys()).rejects.toThrow('Token expired');
    });
  });

  describe('apiKeyApi.revokeKey', () => {
    test('revokes API key by ID', async () => {
      mockLocalStorage.store = { 'jadevectordb_api_key': 'admin-token' };

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, message: 'API key revoked' })
      });

      const result = await apiKeyApi.revokeKey('key-123');

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/v1/apikeys/key-123'),
        expect.objectContaining({
          method: 'DELETE',
          headers: expect.objectContaining({
            'X-API-Key': 'admin-token'
          })
        })
      );

      expect(result.success).toBe(true);
    });

    test('handles non-existent key', async () => {
      mockLocalStorage.store = { 'jadevectordb_api_key': 'test-token' };

      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: async () => ({ message: 'API key not found' })
      });

      await expect(apiKeyApi.revokeKey('nonexistent-key'))
        .rejects.toThrow('API key not found');
    });

    test('prevents revoking without permission', async () => {
      mockLocalStorage.store = { 'jadevectordb_api_key': 'limited-token' };

      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 403,
        json: async () => ({ message: 'Cannot revoke this key' })
      });

      await expect(apiKeyApi.revokeKey('key-123'))
        .rejects.toThrow('Cannot revoke this key');
    });
  });

  describe('apiKeyApi.validateKey', () => {
    test('validates API key', async () => {
      const testKey = 'sk_test_validkey123';

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          valid: true,
          keyId: 'key-123',
          permissions: ['read', 'write']
        })
      });

      const result = await apiKeyApi.validateKey(testKey);

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/v1/apikeys/validate'),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ apiKey: testKey })
        })
      });

      expect(result.valid).toBe(true);
    });

    test('identifies invalid API key', async () => {
      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: async () => ({ valid: false, message: 'Invalid API key' })
      });

      await expect(apiKeyApi.validateKey('invalid-key'))
        .rejects.toThrow('Invalid API key');
    });
  });

  // ============================================================================
  // Request Header Tests
  // ============================================================================

  describe('Auth Headers', () => {
    test('includes API key in X-API-Key header when available', async () => {
      mockLocalStorage.store = { 'jadevectordb_api_key': 'test-api-key-123' };

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ apiKeys: [] })
      });

      await apiKeyApi.listKeys();

      const headers = global.fetch.mock.calls[0][1].headers;
      expect(headers['X-API-Key']).toBe('test-api-key-123');
    });

    test('omits X-API-Key header when not authenticated', async () => {
      mockLocalStorage.store = {}; // No API key

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({})
      });

      await authApi.register('user', 'pass');

      const headers = global.fetch.mock.calls[0][1].headers;
      expect(headers['X-API-Key']).toBeUndefined();
    });

    test('always includes Content-Type header', async () => {
      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({})
      });

      await authApi.login('user', 'pass');

      const headers = global.fetch.mock.calls[0][1].headers;
      expect(headers['Content-Type']).toBe('application/json');
    });
  });

  // ============================================================================
  // Error Response Handling
  // ============================================================================

  describe('Error Response Handling', () => {
    test('extracts error message from response', async () => {
      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => ({ message: 'Custom error message' })
      });

      await expect(authApi.login('user', 'pass'))
        .rejects.toThrow('Custom error message');
    });

    test('uses generic error when no message in response', async () => {
      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: async () => ({}) // No message
      });

      await expect(authApi.login('user', 'pass'))
        .rejects.toThrow('API error: 500');
    });

    test('handles malformed JSON in error response', async () => {
      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: async () => { throw new Error('Invalid JSON'); }
      });

      await expect(authApi.login('user', 'pass'))
        .rejects.toThrow('API error: 500');
    });
  });

  // ============================================================================
  // URL Construction Tests
  // ============================================================================

  describe('API URL Configuration', () => {
    const originalEnv = process.env.NEXT_PUBLIC_API_URL;

    afterEach(() => {
      process.env.NEXT_PUBLIC_API_URL = originalEnv;
    });

    test('uses default URL when environment variable not set', async () => {
      delete process.env.NEXT_PUBLIC_API_URL;

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({})
      });

      await authApi.login('user', 'pass');

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('http://localhost:8080/v1'),
        expect.any(Object)
      );
    });

    test('uses custom URL from environment variable', async () => {
      process.env.NEXT_PUBLIC_API_URL = 'https://api.example.com/v1';

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({})
      });

      await authApi.login('user', 'pass');

      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('https://api.example.com/v1'),
        expect.any(Object)
      );
    });
  });
});
