// frontend/tests/unit/services/auth-api.test.js
// Tests for authApi and apiKeyApi - aligned with actual implementation in src/lib/api.js
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

// Suppress console.warn for "No auth token found" messages
const originalWarn = console.warn;
beforeAll(() => {
  console.warn = jest.fn();
});
afterAll(() => {
  console.warn = originalWarn;
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
        user_id: 'user-1',
        username: 'testuser'
      };

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse
      });

      const result = await authApi.login('testuser', 'password123');

      // Actual implementation uses /api/auth/login (proxied by Next.js)
      expect(global.fetch).toHaveBeenCalledWith(
        '/api/auth/login',
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

    test('stores token and user info in localStorage on successful login', async () => {
      const mockResponse = {
        token: 'jwt-token-123',
        user_id: 'user-1',
        username: 'testuser'
      };

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse
      });

      await authApi.login('testuser', 'password123');

      expect(mockLocalStorage.setItem).toHaveBeenCalledWith('jadevectordb_auth_token', 'jwt-token-123');
      expect(mockLocalStorage.setItem).toHaveBeenCalledWith('jadevectordb_user_id', 'user-1');
      expect(mockLocalStorage.setItem).toHaveBeenCalledWith('jadevectordb_username', 'testuser');
    });

    test('stores must_change_password flag when required', async () => {
      const mockResponse = {
        token: 'jwt-token-123',
        user_id: 'user-1',
        must_change_password: true
      };

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse
      });

      await authApi.login('testuser', 'password123');

      expect(mockLocalStorage.setItem).toHaveBeenCalledWith('jadevectordb_must_change_password', 'true');
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

      // Actual API signature: register(username, password, email, roles)
      const result = await authApi.register('newuser', 'securepass123', 'user@example.com', ['user']);

      expect(global.fetch).toHaveBeenCalledWith(
        '/api/auth/register',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            username: 'newuser',
            password: 'securepass123',
            email: 'user@example.com',
            roles: ['user']
          })
        })
      );

      expect(result).toEqual(mockResponse);
    });

    test('uses default values when optional params not provided', async () => {
      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ userId: 'id' })
      });

      await authApi.register('user', 'pass');

      const callArgs = global.fetch.mock.calls[0][1].body;
      const parsedBody = JSON.parse(callArgs);
      expect(parsedBody.email).toBe('');
      expect(parsedBody.roles).toEqual([]);
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
  });

  describe('authApi.logout', () => {
    test('sends POST request with auth token', async () => {
      mockLocalStorage.store = { 'jadevectordb_auth_token': 'test-token' };

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true })
      });

      const result = await authApi.logout();

      expect(global.fetch).toHaveBeenCalledWith(
        '/api/auth/logout',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Authorization': 'Bearer test-token'
          })
        })
      );

      expect(result).toEqual({ success: true });
    });

    test('throws error when no auth token', async () => {
      mockLocalStorage.store = {}; // No auth token

      await expect(authApi.logout()).rejects.toThrow('No auth token found');
    });

    test('clears localStorage on logout', async () => {
      mockLocalStorage.store = { 'jadevectordb_auth_token': 'test-token' };

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true })
      });

      await authApi.logout();

      expect(mockLocalStorage.removeItem).toHaveBeenCalledWith('jadevectordb_auth_token');
      expect(mockLocalStorage.removeItem).toHaveBeenCalledWith('jadevectordb_user_id');
      expect(mockLocalStorage.removeItem).toHaveBeenCalledWith('jadevectordb_username');
    });
  });

  // ============================================================================
  // API Key Management Tests
  // ============================================================================

  describe('apiKeyApi.createKey', () => {
    test('creates API key with data', async () => {
      mockLocalStorage.store = { 'jadevectordb_auth_token': 'admin-token' };

      const mockResponse = {
        apiKey: 'sk_test_1234567890',
        keyId: 'key-123',
        name: 'Production Key'
      };

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse
      });

      const keyData = {
        name: 'Production Key',
        permissions: ['read', 'write']
      };

      const result = await apiKeyApi.createKey(keyData);

      expect(global.fetch).toHaveBeenCalledWith(
        '/api/apikeys',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Authorization': 'Bearer admin-token'
          }),
          body: JSON.stringify(keyData)
        })
      );

      expect(result).toEqual(mockResponse);
    });

    test('handles permission denied', async () => {
      mockLocalStorage.store = { 'jadevectordb_auth_token': 'limited-token' };

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
      mockLocalStorage.store = { 'jadevectordb_auth_token': 'test-token' };

      const mockResponse = {
        apiKeys: [
          { keyId: 'key-1', name: 'Key 1' },
          { keyId: 'key-2', name: 'Key 2' }
        ],
        total: 2
      };

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse
      });

      const result = await apiKeyApi.listKeys();

      expect(global.fetch).toHaveBeenCalledWith(
        '/api/apikeys',
        expect.objectContaining({
          method: 'GET',
          headers: expect.objectContaining({
            'Authorization': 'Bearer test-token'
          })
        })
      );

      expect(result).toEqual(mockResponse);
      expect(result.apiKeys).toHaveLength(2);
    });

    test('returns empty array when no keys exist', async () => {
      mockLocalStorage.store = { 'jadevectordb_auth_token': 'test-token' };

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ apiKeys: [], total: 0 })
      });

      const result = await apiKeyApi.listKeys();

      expect(result.apiKeys).toEqual([]);
      expect(result.total).toBe(0);
    });
  });

  describe('apiKeyApi.revokeKey', () => {
    test('revokes API key by ID', async () => {
      mockLocalStorage.store = { 'jadevectordb_auth_token': 'admin-token' };

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, message: 'API key revoked' })
      });

      const result = await apiKeyApi.revokeKey('key-123');

      expect(global.fetch).toHaveBeenCalledWith(
        '/api/apikeys/key-123',
        expect.objectContaining({
          method: 'DELETE',
          headers: expect.objectContaining({
            'Authorization': 'Bearer admin-token'
          })
        })
      );

      expect(result.success).toBe(true);
    });

    test('handles non-existent key', async () => {
      mockLocalStorage.store = { 'jadevectordb_auth_token': 'test-token' };

      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: async () => ({ message: 'API key not found' })
      });

      await expect(apiKeyApi.revokeKey('nonexistent-key'))
        .rejects.toThrow('API key not found');
    });
  });

  // ============================================================================
  // Request Header Tests
  // ============================================================================

  describe('Auth Headers', () => {
    test('includes Authorization Bearer token when available', async () => {
      mockLocalStorage.store = { 'jadevectordb_auth_token': 'test-api-key-123' };

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ apiKeys: [] })
      });

      await apiKeyApi.listKeys();

      const headers = global.fetch.mock.calls[0][1].headers;
      expect(headers['Authorization']).toBe('Bearer test-api-key-123');
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
        json: async () => ({})
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
    test('uses /api prefix by default (proxied by Next.js)', async () => {
      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({})
      });

      await authApi.login('user', 'pass');

      expect(global.fetch).toHaveBeenCalledWith(
        '/api/auth/login',
        expect.any(Object)
      );
    });
  });
});
