/**
 * User Management API Tests (T264)
 *
 * Comprehensive test suite for user management API functions.
 * Tests all user management operations with mocked axios calls.
 */

const axios = require('axios');
const {
  createUser,
  listUsers,
  getUser,
  updateUser,
  deleteUser,
  activateUser,
  deactivateUser
} = require('./api');

// Mock axios to prevent actual API calls
jest.mock('axios');
const mockedAxios = axios;

describe('User Management API', () => {
  const mockBaseUrl = 'http://localhost:8080';
  const mockApiKey = 'test-api-key-12345';
  const mockEmail = 'test@example.com';

  let mockApiClient;

  beforeEach(() => {
    jest.clearAllMocks();

    // Setup mock API client
    mockApiClient = {
      post: jest.fn(),
      get: jest.fn(),
      put: jest.fn(),
      delete: jest.fn()
    };

    mockedAxios.create = jest.fn().mockReturnValue(mockApiClient);
  });

  describe('createUser', () => {
    it('should create a user with email, role, and password', async () => {
      const mockResponse = {
        data: {
          user_id: 'user-123',
          email: mockEmail,
          role: 'developer',
          message: 'User created successfully'
        }
      };
      mockApiClient.post.mockResolvedValue(mockResponse);

      const result = await createUser(
        mockBaseUrl,
        mockApiKey,
        mockEmail,
        'developer',
        'password123'
      );

      expect(mockedAxios.create).toHaveBeenCalledWith({
        baseURL: mockBaseUrl,
        timeout: 30000,
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${mockApiKey}`
        }
      });

      expect(mockApiClient.post).toHaveBeenCalledWith('/api/v1/users', {
        email: mockEmail,
        role: 'developer',
        password: 'password123'
      });

      expect(result).toEqual(mockResponse.data);
    });

    it('should create a user without password', async () => {
      const mockResponse = {
        data: { user_id: 'user-456', email: 'admin@example.com' }
      };
      mockApiClient.post.mockResolvedValue(mockResponse);

      await createUser(mockBaseUrl, mockApiKey, 'admin@example.com', 'admin');

      const callArgs = mockApiClient.post.mock.calls[0][1];
      expect(callArgs.email).toBe('admin@example.com');
      expect(callArgs.role).toBe('admin');
      expect(callArgs.password).toBeUndefined();
    });

    it('should handle user already exists error', async () => {
      const mockError = {
        response: {
          status: 409,
          data: { message: 'User already exists' }
        }
      };
      mockApiClient.post.mockRejectedValue(mockError);

      await expect(
        createUser(mockBaseUrl, mockApiKey, mockEmail, 'developer', 'pass')
      ).rejects.toThrow('Failed to create user');
    });

    it('should handle validation errors', async () => {
      const mockError = {
        response: {
          status: 400,
          data: { message: 'Invalid email format' }
        }
      };
      mockApiClient.post.mockRejectedValue(mockError);

      await expect(
        createUser(mockBaseUrl, mockApiKey, 'invalid-email', 'developer')
      ).rejects.toThrow();
    });
  });

  describe('listUsers', () => {
    it('should list all users', async () => {
      const mockResponse = {
        data: {
          users: [
            { user_id: '1', email: 'admin@example.com', role: 'admin', status: 'active' },
            { user_id: '2', email: 'dev@example.com', role: 'developer', status: 'active' }
          ],
          count: 2
        }
      };
      mockApiClient.get.mockResolvedValue(mockResponse);

      const result = await listUsers(mockBaseUrl, mockApiKey);

      expect(mockApiClient.get).toHaveBeenCalledWith('/api/v1/users', { params: {} });
      expect(result).toEqual(mockResponse.data);
      expect(result.users).toHaveLength(2);
    });

    it('should list users with role filter', async () => {
      const mockResponse = {
        data: {
          users: [{ email: 'admin@example.com', role: 'admin' }],
          count: 1
        }
      };
      mockApiClient.get.mockResolvedValue(mockResponse);

      const result = await listUsers(mockBaseUrl, mockApiKey, 'admin', null);

      expect(mockApiClient.get).toHaveBeenCalledWith('/api/v1/users', {
        params: { role: 'admin' }
      });
      expect(result.users[0].role).toBe('admin');
    });

    it('should list users with status filter', async () => {
      const mockResponse = {
        data: {
          users: [{ email: 'test@example.com', status: 'active' }],
          count: 1
        }
      };
      mockApiClient.get.mockResolvedValue(mockResponse);

      await listUsers(mockBaseUrl, mockApiKey, null, 'active');

      expect(mockApiClient.get).toHaveBeenCalledWith('/api/v1/users', {
        params: { status: 'active' }
      });
    });

    it('should list users with both role and status filters', async () => {
      const mockResponse = { data: { users: [], count: 0 } };
      mockApiClient.get.mockResolvedValue(mockResponse);

      await listUsers(mockBaseUrl, mockApiKey, 'developer', 'inactive');

      expect(mockApiClient.get).toHaveBeenCalledWith('/api/v1/users', {
        params: { role: 'developer', status: 'inactive' }
      });
    });

    it('should handle empty user list', async () => {
      const mockResponse = { data: { users: [], count: 0 } };
      mockApiClient.get.mockResolvedValue(mockResponse);

      const result = await listUsers(mockBaseUrl, mockApiKey);

      expect(result.users).toHaveLength(0);
      expect(result.count).toBe(0);
    });
  });

  describe('getUser', () => {
    it('should get user details by email', async () => {
      const mockResponse = {
        data: {
          user_id: 'user-123',
          email: mockEmail,
          role: 'developer',
          status: 'active',
          created_at: '2024-01-01T00:00:00Z'
        }
      };
      mockApiClient.get.mockResolvedValue(mockResponse);

      const result = await getUser(mockBaseUrl, mockApiKey, mockEmail);

      expect(mockApiClient.get).toHaveBeenCalledWith(`/api/v1/users/${mockEmail}`);
      expect(result.email).toBe(mockEmail);
      expect(result.role).toBe('developer');
    });

    it('should handle user not found error', async () => {
      const mockError = {
        response: {
          status: 404,
          data: { message: 'User not found' }
        }
      };
      mockApiClient.get.mockRejectedValue(mockError);

      await expect(
        getUser(mockBaseUrl, mockApiKey, 'nonexistent@example.com')
      ).rejects.toThrow('Failed to get user');
    });
  });

  describe('updateUser', () => {
    it('should update user role', async () => {
      const mockResponse = {
        data: {
          user_id: 'user-123',
          email: mockEmail,
          role: 'admin',
          message: 'User updated successfully'
        }
      };
      mockApiClient.put.mockResolvedValue(mockResponse);

      const result = await updateUser(mockBaseUrl, mockApiKey, mockEmail, 'admin', null);

      expect(mockApiClient.put).toHaveBeenCalledWith(`/api/v1/users/${mockEmail}`, {
        role: 'admin'
      });
      expect(result.role).toBe('admin');
    });

    it('should update user status', async () => {
      const mockResponse = {
        data: { email: mockEmail, status: 'inactive' }
      };
      mockApiClient.put.mockResolvedValue(mockResponse);

      await updateUser(mockBaseUrl, mockApiKey, mockEmail, null, 'inactive');

      expect(mockApiClient.put).toHaveBeenCalledWith(`/api/v1/users/${mockEmail}`, {
        status: 'inactive'
      });
    });

    it('should update both role and status', async () => {
      const mockResponse = {
        data: { email: mockEmail, role: 'viewer', status: 'active' }
      };
      mockApiClient.put.mockResolvedValue(mockResponse);

      await updateUser(mockBaseUrl, mockApiKey, mockEmail, 'viewer', 'active');

      expect(mockApiClient.put).toHaveBeenCalledWith(`/api/v1/users/${mockEmail}`, {
        role: 'viewer',
        status: 'active'
      });
    });

    it('should handle unauthorized update error', async () => {
      const mockError = {
        response: {
          status: 403,
          data: { message: 'Permission denied' }
        }
      };
      mockApiClient.put.mockRejectedValue(mockError);

      await expect(
        updateUser(mockBaseUrl, mockApiKey, mockEmail, 'admin')
      ).rejects.toThrow();
    });
  });

  describe('deleteUser', () => {
    it('should delete a user', async () => {
      const mockResponse = {
        data: { message: 'User deleted successfully' }
      };
      mockApiClient.delete.mockResolvedValue(mockResponse);

      const result = await deleteUser(mockBaseUrl, mockApiKey, mockEmail);

      expect(mockApiClient.delete).toHaveBeenCalledWith(`/api/v1/users/${mockEmail}`);
      expect(result.message).toBe('User deleted successfully');
    });

    it('should handle delete non-existent user error', async () => {
      const mockError = {
        response: {
          status: 404,
          data: { message: 'User not found' }
        }
      };
      mockApiClient.delete.mockRejectedValue(mockError);

      await expect(
        deleteUser(mockBaseUrl, mockApiKey, 'nonexistent@example.com')
      ).rejects.toThrow('Failed to delete user');
    });

    it('should handle permission denied error', async () => {
      const mockError = {
        response: {
          status: 403,
          data: { message: 'Cannot delete admin user' }
        }
      };
      mockApiClient.delete.mockRejectedValue(mockError);

      await expect(
        deleteUser(mockBaseUrl, mockApiKey, 'admin@example.com')
      ).rejects.toThrow();
    });
  });

  describe('activateUser', () => {
    it('should activate a user', async () => {
      const mockResponse = {
        data: {
          user_id: 'user-123',
          email: mockEmail,
          status: 'active',
          message: 'User activated successfully'
        }
      };
      mockApiClient.put.mockResolvedValue(mockResponse);

      const result = await activateUser(mockBaseUrl, mockApiKey, mockEmail);

      expect(mockApiClient.put).toHaveBeenCalledWith(`/api/v1/users/${mockEmail}/activate`);
      expect(result.status).toBe('active');
    });

    it('should handle already active user', async () => {
      const mockResponse = {
        data: { email: mockEmail, status: 'active', message: 'User is already active' }
      };
      mockApiClient.put.mockResolvedValue(mockResponse);

      const result = await activateUser(mockBaseUrl, mockApiKey, mockEmail);

      expect(result.status).toBe('active');
    });
  });

  describe('deactivateUser', () => {
    it('should deactivate a user', async () => {
      const mockResponse = {
        data: {
          user_id: 'user-123',
          email: mockEmail,
          status: 'inactive',
          message: 'User deactivated successfully'
        }
      };
      mockApiClient.put.mockResolvedValue(mockResponse);

      const result = await deactivateUser(mockBaseUrl, mockApiKey, mockEmail);

      expect(mockApiClient.put).toHaveBeenCalledWith(`/api/v1/users/${mockEmail}/deactivate`);
      expect(result.status).toBe('inactive');
    });

    it('should handle already inactive user', async () => {
      const mockResponse = {
        data: { email: mockEmail, status: 'inactive' }
      };
      mockApiClient.put.mockResolvedValue(mockResponse);

      const result = await deactivateUser(mockBaseUrl, mockApiKey, mockEmail);

      expect(result.status).toBe('inactive');
    });
  });

  describe('User Management Workflow', () => {
    it('should complete full user lifecycle', async () => {
      // 1. Create user
      mockApiClient.post.mockResolvedValueOnce({
        data: { user_id: 'user-999', email: 'lifecycle@example.com' }
      });

      const created = await createUser(
        mockBaseUrl,
        mockApiKey,
        'lifecycle@example.com',
        'developer',
        'password123'
      );
      expect(created.user_id).toBe('user-999');

      // 2. List users (verify user exists)
      mockApiClient.get.mockResolvedValueOnce({
        data: {
          users: [{ email: 'lifecycle@example.com', status: 'active' }],
          count: 1
        }
      });

      const users = await listUsers(mockBaseUrl, mockApiKey);
      expect(users.users).toHaveLength(1);

      // 3. Update user role
      mockApiClient.put.mockResolvedValueOnce({
        data: { email: 'lifecycle@example.com', role: 'admin' }
      });

      await updateUser(mockBaseUrl, mockApiKey, 'lifecycle@example.com', 'admin');

      // 4. Deactivate user
      mockApiClient.put.mockResolvedValueOnce({
        data: { email: 'lifecycle@example.com', status: 'inactive' }
      });

      await deactivateUser(mockBaseUrl, mockApiKey, 'lifecycle@example.com');

      // 5. Activate user
      mockApiClient.put.mockResolvedValueOnce({
        data: { email: 'lifecycle@example.com', status: 'active' }
      });

      await activateUser(mockBaseUrl, mockApiKey, 'lifecycle@example.com');

      // 6. Delete user
      mockApiClient.delete.mockResolvedValueOnce({
        data: { message: 'User deleted successfully' }
      });

      const deleted = await deleteUser(mockBaseUrl, mockApiKey, 'lifecycle@example.com');
      expect(deleted.message).toBe('User deleted successfully');

      // Verify all API calls were made
      expect(mockApiClient.post).toHaveBeenCalledTimes(1);
      expect(mockApiClient.get).toHaveBeenCalledTimes(1);
      expect(mockApiClient.put).toHaveBeenCalledTimes(3); // update, deactivate, activate
      expect(mockApiClient.delete).toHaveBeenCalledTimes(1);
    });
  });

  describe('Error Handling', () => {
    it('should handle network errors', async () => {
      const networkError = new Error('Network error');
      mockApiClient.post.mockRejectedValue(networkError);

      await expect(
        createUser(mockBaseUrl, mockApiKey, mockEmail, 'developer')
      ).rejects.toThrow();
    });

    it('should handle timeout errors', async () => {
      const timeoutError = {
        code: 'ECONNABORTED',
        message: 'timeout of 30000ms exceeded'
      };
      mockApiClient.get.mockRejectedValue(timeoutError);

      await expect(
        listUsers(mockBaseUrl, mockApiKey)
      ).rejects.toThrow();
    });

    it('should handle server errors', async () => {
      const serverError = {
        response: {
          status: 500,
          data: { message: 'Internal server error' }
        }
      };
      mockApiClient.post.mockRejectedValue(serverError);

      await expect(
        createUser(mockBaseUrl, mockApiKey, mockEmail, 'developer')
      ).rejects.toThrow();
    });
  });
});
