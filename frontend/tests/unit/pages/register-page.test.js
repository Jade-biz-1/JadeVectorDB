// frontend/tests/unit/pages/register-page.test.js
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import Register from '@/pages/register';

// Mock the API functions
jest.mock('@/lib/api', () => ({
  authApi: {
    register: jest.fn(),
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

describe('Register Page', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();

    // Mock successful registration
    authApi.register.mockResolvedValue({
      user_id: 'new-user-123',
      username: 'testuser'
    });
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('Rendering', () => {
    test('renders page title', () => {
      render(<Register />);
      expect(screen.getByText('JadeVectorDB')).toBeInTheDocument();
    });

    test('renders sign up card title', () => {
      render(<Register />);
      expect(screen.getByText('Sign Up')).toBeInTheDocument();
    });

    test('renders card description', () => {
      render(<Register />);
      expect(screen.getByText(/Create a new account to get started/)).toBeInTheDocument();
    });

    test('renders username input', () => {
      render(<Register />);
      expect(screen.getByLabelText(/username/i)).toBeInTheDocument();
    });

    test('renders email input', () => {
      render(<Register />);
      expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    });

    test('renders password input', () => {
      render(<Register />);
      expect(screen.getByLabelText(/^password/i)).toBeInTheDocument();
    });

    test('renders confirm password input', () => {
      render(<Register />);
      expect(screen.getByLabelText(/confirm password/i)).toBeInTheDocument();
    });

    test('renders create account button', () => {
      render(<Register />);
      expect(screen.getByRole('button', { name: /create account/i })).toBeInTheDocument();
    });

    test('renders sign in link', () => {
      render(<Register />);
      expect(screen.getByText(/sign in/i)).toBeInTheDocument();
    });

    test('renders back to home link', () => {
      render(<Register />);
      expect(screen.getByText(/back to home/i)).toBeInTheDocument();
    });

    test('shows username requirements hint', () => {
      render(<Register />);
      expect(screen.getByText(/at least 3 characters/i)).toBeInTheDocument();
    });

    test('shows password requirements hint', () => {
      render(<Register />);
      expect(screen.getByText(/at least 8 characters/i)).toBeInTheDocument();
    });
  });

  describe('Form Validation', () => {
    test('shows error for short username', async () => {
      render(<Register />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'ab' } });
      fireEvent.change(screen.getByLabelText(/^password/i), { target: { value: 'password123' } });
      fireEvent.change(screen.getByLabelText(/confirm password/i), { target: { value: 'password123' } });

      fireEvent.click(screen.getByRole('button', { name: /create account/i }));

      await waitFor(() => {
        expect(screen.getByText(/username must be at least 3 characters/i)).toBeInTheDocument();
      });

      expect(authApi.register).not.toHaveBeenCalled();
    });

    test('shows error for short password', async () => {
      render(<Register />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      fireEvent.change(screen.getByLabelText(/^password/i), { target: { value: 'short' } });
      fireEvent.change(screen.getByLabelText(/confirm password/i), { target: { value: 'short' } });

      fireEvent.click(screen.getByRole('button', { name: /create account/i }));

      await waitFor(() => {
        expect(screen.getByText(/password must be at least 8 characters/i)).toBeInTheDocument();
      });

      expect(authApi.register).not.toHaveBeenCalled();
    });

    test('shows error when passwords do not match', async () => {
      render(<Register />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      fireEvent.change(screen.getByLabelText(/^password/i), { target: { value: 'password123' } });
      fireEvent.change(screen.getByLabelText(/confirm password/i), { target: { value: 'different456' } });

      fireEvent.click(screen.getByRole('button', { name: /create account/i }));

      await waitFor(() => {
        expect(screen.getByText(/passwords do not match/i)).toBeInTheDocument();
      });

      expect(authApi.register).not.toHaveBeenCalled();
    });

    // Skip: HTML5 email input validation prevents form submission with invalid email
    test.skip('shows error for invalid email format', async () => {
      render(<Register />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      fireEvent.change(screen.getByLabelText(/email/i), { target: { value: 'invalidemail' } });
      fireEvent.change(screen.getByLabelText(/^password/i), { target: { value: 'password123' } });
      fireEvent.change(screen.getByLabelText(/confirm password/i), { target: { value: 'password123' } });

      fireEvent.click(screen.getByRole('button', { name: /create account/i }));

      await waitFor(() => {
        expect(screen.getByText(/please enter a valid email address/i)).toBeInTheDocument();
      });

      expect(authApi.register).not.toHaveBeenCalled();
    });

    test('accepts valid email', async () => {
      render(<Register />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      fireEvent.change(screen.getByLabelText(/email/i), { target: { value: 'test@example.com' } });
      fireEvent.change(screen.getByLabelText(/^password/i), { target: { value: 'password123' } });
      fireEvent.change(screen.getByLabelText(/confirm password/i), { target: { value: 'password123' } });

      fireEvent.click(screen.getByRole('button', { name: /create account/i }));

      await waitFor(() => {
        expect(authApi.register).toHaveBeenCalled();
      });
    });

    test('allows registration without email (optional)', async () => {
      render(<Register />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      // Don't fill email - it's optional
      fireEvent.change(screen.getByLabelText(/^password/i), { target: { value: 'password123' } });
      fireEvent.change(screen.getByLabelText(/confirm password/i), { target: { value: 'password123' } });

      fireEvent.click(screen.getByRole('button', { name: /create account/i }));

      await waitFor(() => {
        expect(authApi.register).toHaveBeenCalled();
      });
    });
  });

  describe('Form Submission', () => {
    test('calls register API with credentials', async () => {
      render(<Register />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'newuser' } });
      fireEvent.change(screen.getByLabelText(/email/i), { target: { value: 'newuser@example.com' } });
      fireEvent.change(screen.getByLabelText(/^password/i), { target: { value: 'securepassword123' } });
      fireEvent.change(screen.getByLabelText(/confirm password/i), { target: { value: 'securepassword123' } });

      fireEvent.click(screen.getByRole('button', { name: /create account/i }));

      await waitFor(() => {
        expect(authApi.register).toHaveBeenCalledWith(
          'newuser',
          'securepassword123',
          'newuser@example.com',
          [] // Default empty roles
        );
      });
    });

    test('shows loading state during registration', async () => {
      authApi.register.mockImplementation(() => new Promise(() => {}));

      render(<Register />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      fireEvent.change(screen.getByLabelText(/^password/i), { target: { value: 'password123' } });
      fireEvent.change(screen.getByLabelText(/confirm password/i), { target: { value: 'password123' } });

      fireEvent.click(screen.getByRole('button', { name: /create account/i }));

      expect(screen.getByRole('button', { name: /creating account/i })).toBeInTheDocument();
    });

    test('shows success message on successful registration', async () => {
      render(<Register />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      fireEvent.change(screen.getByLabelText(/^password/i), { target: { value: 'password123' } });
      fireEvent.change(screen.getByLabelText(/confirm password/i), { target: { value: 'password123' } });

      fireEvent.click(screen.getByRole('button', { name: /create account/i }));

      await waitFor(() => {
        expect(screen.getByText(/account created successfully/i)).toBeInTheDocument();
      });
    });

    test('shows user ID in success message', async () => {
      render(<Register />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      fireEvent.change(screen.getByLabelText(/^password/i), { target: { value: 'password123' } });
      fireEvent.change(screen.getByLabelText(/confirm password/i), { target: { value: 'password123' } });

      fireEvent.click(screen.getByRole('button', { name: /create account/i }));

      await waitFor(() => {
        expect(screen.getByText(/new-user-123/)).toBeInTheDocument();
      });
    });

    test('redirects to login after successful registration', async () => {
      render(<Register />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      fireEvent.change(screen.getByLabelText(/^password/i), { target: { value: 'password123' } });
      fireEvent.change(screen.getByLabelText(/confirm password/i), { target: { value: 'password123' } });

      fireEvent.click(screen.getByRole('button', { name: /create account/i }));

      await waitFor(() => {
        expect(screen.getByText(/account created successfully/i)).toBeInTheDocument();
      });

      // Fast-forward timer for redirect
      jest.advanceTimersByTime(2000);

      expect(mockPush).toHaveBeenCalledWith('/login');
    });
  });

  describe('Error Handling', () => {
    test('shows error on registration failure', async () => {
      authApi.register.mockRejectedValue(new Error('Username already exists'));

      render(<Register />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'existinguser' } });
      fireEvent.change(screen.getByLabelText(/^password/i), { target: { value: 'password123' } });
      fireEvent.change(screen.getByLabelText(/confirm password/i), { target: { value: 'password123' } });

      fireEvent.click(screen.getByRole('button', { name: /create account/i }));

      await waitFor(() => {
        expect(screen.getByText(/username already exists/i)).toBeInTheDocument();
      });
    });

    test('shows generic error when no message provided', async () => {
      authApi.register.mockRejectedValue(new Error());

      render(<Register />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      fireEvent.change(screen.getByLabelText(/^password/i), { target: { value: 'password123' } });
      fireEvent.change(screen.getByLabelText(/confirm password/i), { target: { value: 'password123' } });

      fireEvent.click(screen.getByRole('button', { name: /create account/i }));

      await waitFor(() => {
        expect(screen.getByText(/registration failed/i)).toBeInTheDocument();
      });
    });
  });

  describe('Form Interactions', () => {
    test('disables inputs during loading', async () => {
      authApi.register.mockImplementation(() => new Promise(() => {}));

      render(<Register />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      fireEvent.change(screen.getByLabelText(/^password/i), { target: { value: 'password123' } });
      fireEvent.change(screen.getByLabelText(/confirm password/i), { target: { value: 'password123' } });

      fireEvent.click(screen.getByRole('button', { name: /create account/i }));

      expect(screen.getByLabelText(/username/i)).toBeDisabled();
      expect(screen.getByLabelText(/email/i)).toBeDisabled();
      expect(screen.getByLabelText(/^password/i)).toBeDisabled();
      expect(screen.getByLabelText(/confirm password/i)).toBeDisabled();
    });

    test('disables submit button during loading', async () => {
      authApi.register.mockImplementation(() => new Promise(() => {}));

      render(<Register />);

      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'testuser' } });
      fireEvent.change(screen.getByLabelText(/^password/i), { target: { value: 'password123' } });
      fireEvent.change(screen.getByLabelText(/confirm password/i), { target: { value: 'password123' } });

      fireEvent.click(screen.getByRole('button', { name: /create account/i }));

      expect(screen.getByRole('button', { name: /creating account/i })).toBeDisabled();
    });

    test('clears error when form values change', async () => {
      render(<Register />);

      // Trigger validation error
      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'ab' } });
      fireEvent.change(screen.getByLabelText(/^password/i), { target: { value: 'password123' } });
      fireEvent.change(screen.getByLabelText(/confirm password/i), { target: { value: 'password123' } });
      fireEvent.click(screen.getByRole('button', { name: /create account/i }));

      await waitFor(() => {
        expect(screen.getByText(/username must be at least 3 characters/i)).toBeInTheDocument();
      });

      // Fix the username and try again
      fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'validuser' } });
      fireEvent.click(screen.getByRole('button', { name: /create account/i }));

      await waitFor(() => {
        expect(screen.getByText(/account created successfully/i)).toBeInTheDocument();
      });

      // Error should be cleared
      expect(screen.queryByText(/username must be at least 3 characters/i)).not.toBeInTheDocument();
    });
  });
});
