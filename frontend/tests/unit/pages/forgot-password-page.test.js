// frontend/tests/unit/pages/forgot-password-page.test.js
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import ForgotPassword from '@/pages/forgot-password';

// Mock the API functions
jest.mock('@/lib/api', () => ({
  authApi: {
    forgotPassword: jest.fn(),
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
jest.mock('next/router', () => ({
  useRouter: () => ({
    query: {},
    push: jest.fn(),
  })
}));

describe('Forgot Password Page', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    // Mock successful forgot password response
    authApi.forgotPassword.mockResolvedValue({
      reset_token: 'reset-token-abc123',
      user_id: 'user-123'
    });
  });

  describe('Rendering', () => {
    test('renders page title', () => {
      render(<ForgotPassword />);
      expect(screen.getByText('JadeVectorDB')).toBeInTheDocument();
    });

    test('renders forgot password card title', () => {
      render(<ForgotPassword />);
      expect(screen.getByText('Forgot Password')).toBeInTheDocument();
    });

    test('renders card description', () => {
      render(<ForgotPassword />);
      expect(screen.getByText(/Enter your username or email to receive password reset instructions/)).toBeInTheDocument();
    });

    test('renders username input', () => {
      render(<ForgotPassword />);
      expect(screen.getByLabelText(/username/i)).toBeInTheDocument();
    });

    test('renders email input', () => {
      render(<ForgotPassword />);
      expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    });

    test('renders OR separator', () => {
      render(<ForgotPassword />);
      expect(screen.getByText('OR')).toBeInTheDocument();
    });

    test('renders submit button', () => {
      render(<ForgotPassword />);
      expect(screen.getByRole('button', { name: /send reset instructions/i })).toBeInTheDocument();
    });

    test('renders sign in link', () => {
      render(<ForgotPassword />);
      expect(screen.getByText(/sign in/i)).toBeInTheDocument();
    });

    test('renders reset token link', () => {
      render(<ForgotPassword />);
      expect(screen.getByText(/already have a reset token/i)).toBeInTheDocument();
    });

    test('renders back to home link', () => {
      render(<ForgotPassword />);
      expect(screen.getByText(/back to home/i)).toBeInTheDocument();
    });
  });

  describe('Form Validation', () => {
    test('shows error when both fields empty', async () => {
      render(<ForgotPassword />);

      fireEvent.click(screen.getByRole('button', { name: /send reset instructions/i }));

      await waitFor(() => {
        expect(screen.getByText(/please enter either your username or email/i)).toBeInTheDocument();
      });

      expect(authApi.forgotPassword).not.toHaveBeenCalled();
    });

    test('accepts username only', async () => {
      render(<ForgotPassword />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      fireEvent.click(screen.getByRole('button', { name: /send reset instructions/i }));

      await waitFor(() => {
        expect(authApi.forgotPassword).toHaveBeenCalledWith('testuser', '');
      });
    });

    test('accepts email only', async () => {
      render(<ForgotPassword />);

      fireEvent.change(screen.getByLabelText(/email/i), { target: { value: 'test@example.com' } });
      fireEvent.click(screen.getByRole('button', { name: /send reset instructions/i }));

      await waitFor(() => {
        expect(authApi.forgotPassword).toHaveBeenCalledWith('', 'test@example.com');
      });
    });

    test('accepts both username and email', async () => {
      render(<ForgotPassword />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      fireEvent.change(screen.getByLabelText(/email/i), { target: { value: 'test@example.com' } });
      fireEvent.click(screen.getByRole('button', { name: /send reset instructions/i }));

      await waitFor(() => {
        expect(authApi.forgotPassword).toHaveBeenCalledWith('testuser', 'test@example.com');
      });
    });
  });

  describe('Form Submission', () => {
    test('calls forgotPassword API with credentials', async () => {
      render(<ForgotPassword />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      fireEvent.click(screen.getByRole('button', { name: /send reset instructions/i }));

      await waitFor(() => {
        expect(authApi.forgotPassword).toHaveBeenCalledWith('testuser', '');
      });
    });

    test('shows loading state during submission', async () => {
      authApi.forgotPassword.mockImplementation(() => new Promise(() => {}));

      render(<ForgotPassword />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      fireEvent.click(screen.getByRole('button', { name: /send reset instructions/i }));

      expect(screen.getByRole('button', { name: /processing/i })).toBeInTheDocument();
    });

    test('shows success message with reset token', async () => {
      render(<ForgotPassword />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      fireEvent.click(screen.getByRole('button', { name: /send reset instructions/i }));

      await waitFor(() => {
        expect(screen.getByText(/password reset initiated/i)).toBeInTheDocument();
        expect(screen.getByText(/reset-token-abc123/)).toBeInTheDocument();
        expect(screen.getByText(/user-123/)).toBeInTheDocument();
      });
    });

    test('shows success message without token when not provided', async () => {
      authApi.forgotPassword.mockResolvedValue({});

      render(<ForgotPassword />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      fireEvent.click(screen.getByRole('button', { name: /send reset instructions/i }));

      await waitFor(() => {
        expect(screen.getByText(/password reset instructions have been sent/i)).toBeInTheDocument();
      });
    });

    test('clears form after successful submission', async () => {
      render(<ForgotPassword />);

      const usernameInput = screen.getByLabelText(/username/i);
      const emailInput = screen.getByLabelText(/email/i);

      fireEvent.change(usernameInput, { target: { value: 'testuser' } });
      fireEvent.change(emailInput, { target: { value: 'test@example.com' } });
      fireEvent.click(screen.getByRole('button', { name: /send reset instructions/i }));

      await waitFor(() => {
        expect(screen.getByText(/reset-token-abc123/)).toBeInTheDocument();
      });

      expect(usernameInput).toHaveValue('');
      expect(emailInput).toHaveValue('');
    });
  });

  describe('Error Handling', () => {
    test('shows error on API failure', async () => {
      authApi.forgotPassword.mockRejectedValue(new Error('User not found'));

      render(<ForgotPassword />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'nonexistent' } });
      fireEvent.click(screen.getByRole('button', { name: /send reset instructions/i }));

      await waitFor(() => {
        expect(screen.getByText(/user not found/i)).toBeInTheDocument();
      });
    });

    test('shows generic error when no message provided', async () => {
      authApi.forgotPassword.mockRejectedValue(new Error());

      render(<ForgotPassword />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      fireEvent.click(screen.getByRole('button', { name: /send reset instructions/i }));

      await waitFor(() => {
        expect(screen.getByText(/failed to process password reset request/i)).toBeInTheDocument();
      });
    });
  });

  describe('Form Interactions', () => {
    test('disables inputs during loading', async () => {
      authApi.forgotPassword.mockImplementation(() => new Promise(() => {}));

      render(<ForgotPassword />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      fireEvent.click(screen.getByRole('button', { name: /send reset instructions/i }));

      expect(screen.getByLabelText(/username/i)).toBeDisabled();
      expect(screen.getByLabelText(/email/i)).toBeDisabled();
    });

    test('disables submit button during loading', async () => {
      authApi.forgotPassword.mockImplementation(() => new Promise(() => {}));

      render(<ForgotPassword />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      fireEvent.click(screen.getByRole('button', { name: /send reset instructions/i }));

      expect(screen.getByRole('button', { name: /processing/i })).toBeDisabled();
    });

    test('clears error when retrying', async () => {
      authApi.forgotPassword
        .mockRejectedValueOnce(new Error('User not found'))
        .mockResolvedValueOnce({ reset_token: 'token-123', user_id: 'user-123' });

      render(<ForgotPassword />);

      // First attempt - fails
      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'nonexistent' } });
      fireEvent.click(screen.getByRole('button', { name: /send reset instructions/i }));

      await waitFor(() => {
        expect(screen.getByText(/user not found/i)).toBeInTheDocument();
      });

      // Second attempt - should clear error
      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      fireEvent.click(screen.getByRole('button', { name: /send reset instructions/i }));

      await waitFor(() => {
        expect(screen.getByText(/password reset initiated/i)).toBeInTheDocument();
      });

      expect(screen.queryByText(/user not found/i)).not.toBeInTheDocument();
    });
  });
});
