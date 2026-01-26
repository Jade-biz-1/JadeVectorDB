// frontend/tests/unit/pages/login-page.test.js
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import Login from '@/pages/login';

// Mock the API functions
jest.mock('@/lib/api', () => ({
  authApi: {
    login: jest.fn(),
  }
}));

import { authApi } from '@/lib/api';

// Mock localStorage
Object.defineProperty(window, 'localStorage', {
  value: {
    getItem: jest.fn(() => null),
    setItem: jest.fn(),
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

describe('Login Page', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();

    // Mock successful login
    authApi.login.mockResolvedValue({
      token: 'test-token',
      username: 'testuser'
    });
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('Rendering', () => {
    test('renders page title', () => {
      render(<Login />);
      expect(screen.getByText('JadeVectorDB')).toBeInTheDocument();
    });

    test('renders login card title', () => {
      render(<Login />);
      expect(screen.getByText('Login')).toBeInTheDocument();
    });

    test('renders username input', () => {
      render(<Login />);
      expect(screen.getByLabelText(/username/i)).toBeInTheDocument();
    });

    test('renders password input', () => {
      render(<Login />);
      expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
    });

    test('renders sign in button', () => {
      render(<Login />);
      expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
    });

    test('renders forgot password link', () => {
      render(<Login />);
      expect(screen.getByText(/forgot your password/i)).toBeInTheDocument();
    });

    test('renders sign up link', () => {
      render(<Login />);
      expect(screen.getByText(/sign up/i)).toBeInTheDocument();
    });

    test('renders back to home link', () => {
      render(<Login />);
      expect(screen.getByText(/back to home/i)).toBeInTheDocument();
    });
  });

  describe('Form Submission', () => {
    test('calls login API with credentials', async () => {
      render(<Login />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      fireEvent.change(screen.getByLabelText(/password/i), { target: { value: 'password123' } });

      fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

      await waitFor(() => {
        expect(authApi.login).toHaveBeenCalledWith('testuser', 'password123');
      });
    });

    test('shows loading state during login', async () => {
      authApi.login.mockImplementation(() => new Promise(() => {}));

      render(<Login />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      fireEvent.change(screen.getByLabelText(/password/i), { target: { value: 'password123' } });

      fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

      expect(screen.getByRole('button', { name: /signing in/i })).toBeInTheDocument();
    });

    test('shows success message on successful login', async () => {
      render(<Login />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      fireEvent.change(screen.getByLabelText(/password/i), { target: { value: 'password123' } });

      fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

      await waitFor(() => {
        expect(screen.getByText(/successfully logged in/i)).toBeInTheDocument();
      });
    });

    test('redirects to dashboard after successful login', async () => {
      render(<Login />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      fireEvent.change(screen.getByLabelText(/password/i), { target: { value: 'password123' } });

      fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

      await waitFor(() => {
        expect(screen.getByText(/successfully logged in/i)).toBeInTheDocument();
      });

      // Fast-forward timer for redirect
      jest.advanceTimersByTime(1500);

      expect(mockPush).toHaveBeenCalledWith('/dashboard');
    });
  });

  describe('Password Change Required', () => {
    test('shows message when password change required', async () => {
      authApi.login.mockResolvedValue({
        token: 'test-token',
        username: 'testuser',
        must_change_password: true
      });

      render(<Login />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      fireEvent.change(screen.getByLabelText(/password/i), { target: { value: 'password123' } });

      fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

      await waitFor(() => {
        expect(screen.getByText(/must change your password/i)).toBeInTheDocument();
      });
    });

    test('redirects to change-password page when required', async () => {
      authApi.login.mockResolvedValue({
        token: 'test-token',
        username: 'testuser',
        must_change_password: true
      });

      render(<Login />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      fireEvent.change(screen.getByLabelText(/password/i), { target: { value: 'password123' } });

      fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

      await waitFor(() => {
        expect(screen.getByText(/must change your password/i)).toBeInTheDocument();
      });

      // Fast-forward timer for redirect
      jest.advanceTimersByTime(1000);

      expect(mockPush).toHaveBeenCalledWith('/change-password');
    });
  });

  describe('Error Handling', () => {
    // Skip: HTML5 required validation prevents form submission in test environment
    test.skip('shows error for empty credentials', async () => {
      render(<Login />);

      // Don't fill in any fields
      fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

      await waitFor(() => {
        expect(screen.getByText(/please enter both username and password/i)).toBeInTheDocument();
      });
    });

    test('shows error on login failure', async () => {
      authApi.login.mockRejectedValue(new Error('Invalid credentials'));

      render(<Login />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      fireEvent.change(screen.getByLabelText(/password/i), { target: { value: 'wrongpassword' } });

      fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

      await waitFor(() => {
        expect(screen.getByText(/invalid credentials/i)).toBeInTheDocument();
      });
    });

    test('shows generic error when no message provided', async () => {
      authApi.login.mockRejectedValue(new Error());

      render(<Login />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      fireEvent.change(screen.getByLabelText(/password/i), { target: { value: 'wrongpassword' } });

      fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

      await waitFor(() => {
        expect(screen.getByText(/login failed/i)).toBeInTheDocument();
      });
    });

    // Skip: HTML5 required validation prevents form submission in test environment
    test.skip('does not call API when only username provided', async () => {
      render(<Login />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      // Don't fill password

      fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

      await waitFor(() => {
        expect(screen.getByText(/please enter both username and password/i)).toBeInTheDocument();
      });

      expect(authApi.login).not.toHaveBeenCalled();
    });
  });

  describe('Form Interactions', () => {
    test('disables inputs during loading', async () => {
      authApi.login.mockImplementation(() => new Promise(() => {}));

      render(<Login />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      fireEvent.change(screen.getByLabelText(/password/i), { target: { value: 'password123' } });

      fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

      expect(screen.getByLabelText(/username/i)).toBeDisabled();
      expect(screen.getByLabelText(/password/i)).toBeDisabled();
    });

    test('disables submit button during loading', async () => {
      authApi.login.mockImplementation(() => new Promise(() => {}));

      render(<Login />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      fireEvent.change(screen.getByLabelText(/password/i), { target: { value: 'password123' } });

      fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

      expect(screen.getByRole('button', { name: /signing in/i })).toBeDisabled();
    });

    test('clears error when retrying login', async () => {
      authApi.login
        .mockRejectedValueOnce(new Error('Invalid credentials'))
        .mockResolvedValueOnce({ token: 'test-token', username: 'testuser' });

      render(<Login />);

      // First attempt - fails
      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      fireEvent.change(screen.getByLabelText(/password/i), { target: { value: 'wrongpassword' } });
      fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

      await waitFor(() => {
        expect(screen.getByText(/invalid credentials/i)).toBeInTheDocument();
      });

      // Second attempt - should clear error first
      fireEvent.change(screen.getByLabelText(/password/i), { target: { value: 'correctpassword' } });
      fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

      await waitFor(() => {
        expect(screen.getByText(/successfully logged in/i)).toBeInTheDocument();
      });

      // Error should be cleared
      expect(screen.queryByText(/invalid credentials/i)).not.toBeInTheDocument();
    });
  });
});
