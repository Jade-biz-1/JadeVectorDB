// frontend/tests/unit/pages/change-password-page.test.js
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import ChangePassword from '@/pages/change-password';

// Mock the API functions
jest.mock('@/lib/api', () => ({
  authApi: {
    changePassword: jest.fn(),
    logout: jest.fn(),
  }
}));

import { authApi } from '@/lib/api';

// Mock localStorage
const mockLocalStorage = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
};
Object.defineProperty(window, 'localStorage', {
  value: mockLocalStorage,
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

describe('Change Password Page', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();

    // Default - authenticated user without must_change_password
    mockLocalStorage.getItem.mockImplementation((key) => {
      if (key === 'jadevectordb_auth_token') return 'test-token';
      if (key === 'jadevectordb_user_id') return 'user-123';
      if (key === 'jadevectordb_must_change_password') return null;
      return null;
    });

    // Mock successful password change
    authApi.changePassword.mockResolvedValue({});
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('Authentication Check', () => {
    test('redirects to login if not authenticated', () => {
      mockLocalStorage.getItem.mockReturnValue(null);

      render(<ChangePassword />);

      expect(mockPush).toHaveBeenCalledWith('/login');
    });

    test('does not redirect if authenticated', () => {
      render(<ChangePassword />);

      expect(mockPush).not.toHaveBeenCalledWith('/login');
    });

    test('shows must change password warning when required', () => {
      mockLocalStorage.getItem.mockImplementation((key) => {
        if (key === 'jadevectordb_auth_token') return 'test-token';
        if (key === 'jadevectordb_must_change_password') return 'true';
        return null;
      });

      render(<ChangePassword />);

      expect(screen.getByText('Password Change Required')).toBeInTheDocument();
      expect(screen.getByText(/You must change your password before continuing/)).toBeInTheDocument();
    });
  });

  describe('Rendering', () => {
    test('renders page title', () => {
      render(<ChangePassword />);
      expect(screen.getByText('JadeVectorDB')).toBeInTheDocument();
    });

    test('renders card title', () => {
      render(<ChangePassword />);
      // Card title appears in h3, button also has "Change Password"
      expect(screen.getByRole('heading', { name: /change password/i })).toBeInTheDocument();
    });

    test('renders card description', () => {
      render(<ChangePassword />);
      expect(screen.getByText(/Enter your current password and choose a new one/)).toBeInTheDocument();
    });

    test('renders current password input', () => {
      render(<ChangePassword />);
      expect(screen.getByLabelText(/current password/i)).toBeInTheDocument();
    });

    test('renders new password input', () => {
      render(<ChangePassword />);
      expect(screen.getByLabelText(/^new password/i)).toBeInTheDocument();
    });

    test('renders confirm password input', () => {
      render(<ChangePassword />);
      expect(screen.getByLabelText(/confirm new password/i)).toBeInTheDocument();
    });

    test('renders submit button', () => {
      render(<ChangePassword />);
      expect(screen.getByRole('button', { name: /^change password$/i })).toBeInTheDocument();
    });

    test('renders password requirements hint', () => {
      render(<ChangePassword />);
      expect(screen.getByText(/Must be at least 10 characters/)).toBeInTheDocument();
    });

    test('renders cancel and logout button when not must change', () => {
      render(<ChangePassword />);
      expect(screen.getByRole('button', { name: /cancel and logout/i })).toBeInTheDocument();
    });

    test('hides cancel button when must change password', () => {
      mockLocalStorage.getItem.mockImplementation((key) => {
        if (key === 'jadevectordb_auth_token') return 'test-token';
        if (key === 'jadevectordb_must_change_password') return 'true';
        return null;
      });

      render(<ChangePassword />);

      expect(screen.queryByRole('button', { name: /cancel and logout/i })).not.toBeInTheDocument();
    });
  });

  describe('Password Strength Validation', () => {
    test('shows weak message for short password', () => {
      render(<ChangePassword />);

      fireEvent.change(screen.getByLabelText(/^new password/i), { target: { value: 'short' } });

      expect(screen.getByText(/Password must be at least 10 characters/)).toBeInTheDocument();
    });

    test('shows message for missing character types', () => {
      render(<ChangePassword />);

      fireEvent.change(screen.getByLabelText(/^new password/i), { target: { value: 'lowercaseonly1234567' } });

      expect(screen.getByText(/must contain uppercase, lowercase, digit, and special character/i)).toBeInTheDocument();
    });

    test('shows good message for acceptable password', () => {
      render(<ChangePassword />);

      fireEvent.change(screen.getByLabelText(/^new password/i), { target: { value: 'Password1!' } });

      expect(screen.getByText(/Good - Consider using 12\+ characters/)).toBeInTheDocument();
    });

    test('shows strong message for strong password', () => {
      render(<ChangePassword />);

      fireEvent.change(screen.getByLabelText(/^new password/i), { target: { value: 'StrongPassword123!' } });

      expect(screen.getByText('Strong password')).toBeInTheDocument();
    });
  });

  describe('Form Validation', () => {
    test('shows error for password mismatch', async () => {
      render(<ChangePassword />);

      fireEvent.change(screen.getByLabelText(/current password/i), { target: { value: 'oldpassword' } });
      fireEvent.change(screen.getByLabelText(/^new password/i), { target: { value: 'NewPassword123!' } });
      fireEvent.change(screen.getByLabelText(/confirm new password/i), { target: { value: 'DifferentPassword123!' } });

      fireEvent.click(screen.getByRole('button', { name: /^change password$/i }));

      await waitFor(() => {
        expect(screen.getByText(/new passwords do not match/i)).toBeInTheDocument();
      });

      expect(authApi.changePassword).not.toHaveBeenCalled();
    });

    test('shows error when old and new password are same', async () => {
      render(<ChangePassword />);

      fireEvent.change(screen.getByLabelText(/current password/i), { target: { value: 'SamePassword123!' } });
      fireEvent.change(screen.getByLabelText(/^new password/i), { target: { value: 'SamePassword123!' } });
      fireEvent.change(screen.getByLabelText(/confirm new password/i), { target: { value: 'SamePassword123!' } });

      fireEvent.click(screen.getByRole('button', { name: /^change password$/i }));

      await waitFor(() => {
        expect(screen.getByText(/new password must be different from old password/i)).toBeInTheDocument();
      });

      expect(authApi.changePassword).not.toHaveBeenCalled();
    });

    test('shows error for weak password', async () => {
      render(<ChangePassword />);

      fireEvent.change(screen.getByLabelText(/current password/i), { target: { value: 'oldpassword' } });
      fireEvent.change(screen.getByLabelText(/^new password/i), { target: { value: 'weakpassword' } });
      fireEvent.change(screen.getByLabelText(/confirm new password/i), { target: { value: 'weakpassword' } });

      fireEvent.click(screen.getByRole('button', { name: /^change password$/i }));

      await waitFor(() => {
        // Message appears both in strength indicator and error alert
        const messages = screen.getAllByText(/must contain uppercase, lowercase, digit, and special character/i);
        expect(messages.length).toBeGreaterThanOrEqual(1);
      });

      expect(authApi.changePassword).not.toHaveBeenCalled();
    });

    test('shows error when user ID not found', async () => {
      mockLocalStorage.getItem.mockImplementation((key) => {
        if (key === 'jadevectordb_auth_token') return 'test-token';
        if (key === 'jadevectordb_user_id') return null;
        return null;
      });

      render(<ChangePassword />);

      fireEvent.change(screen.getByLabelText(/current password/i), { target: { value: 'oldpassword' } });
      fireEvent.change(screen.getByLabelText(/^new password/i), { target: { value: 'NewPassword123!' } });
      fireEvent.change(screen.getByLabelText(/confirm new password/i), { target: { value: 'NewPassword123!' } });

      fireEvent.click(screen.getByRole('button', { name: /^change password$/i }));

      await waitFor(() => {
        expect(screen.getByText(/user id not found/i)).toBeInTheDocument();
      });

      expect(authApi.changePassword).not.toHaveBeenCalled();
    });
  });

  describe('Form Submission', () => {
    test('calls changePassword API with credentials', async () => {
      render(<ChangePassword />);

      fireEvent.change(screen.getByLabelText(/current password/i), { target: { value: 'oldpassword' } });
      fireEvent.change(screen.getByLabelText(/^new password/i), { target: { value: 'NewPassword123!' } });
      fireEvent.change(screen.getByLabelText(/confirm new password/i), { target: { value: 'NewPassword123!' } });

      fireEvent.click(screen.getByRole('button', { name: /^change password$/i }));

      await waitFor(() => {
        expect(authApi.changePassword).toHaveBeenCalledWith('user-123', 'oldpassword', 'NewPassword123!');
      });
    });

    test('shows loading state during submission', async () => {
      authApi.changePassword.mockImplementation(() => new Promise(() => {}));

      render(<ChangePassword />);

      fireEvent.change(screen.getByLabelText(/current password/i), { target: { value: 'oldpassword' } });
      fireEvent.change(screen.getByLabelText(/^new password/i), { target: { value: 'NewPassword123!' } });
      fireEvent.change(screen.getByLabelText(/confirm new password/i), { target: { value: 'NewPassword123!' } });

      fireEvent.click(screen.getByRole('button', { name: /^change password$/i }));

      expect(screen.getByRole('button', { name: /changing password/i })).toBeInTheDocument();
    });

    test('shows success message on successful change', async () => {
      render(<ChangePassword />);

      fireEvent.change(screen.getByLabelText(/current password/i), { target: { value: 'oldpassword' } });
      fireEvent.change(screen.getByLabelText(/^new password/i), { target: { value: 'NewPassword123!' } });
      fireEvent.change(screen.getByLabelText(/confirm new password/i), { target: { value: 'NewPassword123!' } });

      fireEvent.click(screen.getByRole('button', { name: /^change password$/i }));

      await waitFor(() => {
        expect(screen.getByText(/password changed successfully/i)).toBeInTheDocument();
      });
    });

    test('redirects to dashboard after successful change', async () => {
      render(<ChangePassword />);

      fireEvent.change(screen.getByLabelText(/current password/i), { target: { value: 'oldpassword' } });
      fireEvent.change(screen.getByLabelText(/^new password/i), { target: { value: 'NewPassword123!' } });
      fireEvent.change(screen.getByLabelText(/confirm new password/i), { target: { value: 'NewPassword123!' } });

      fireEvent.click(screen.getByRole('button', { name: /^change password$/i }));

      await waitFor(() => {
        expect(screen.getByText(/password changed successfully/i)).toBeInTheDocument();
      });

      jest.advanceTimersByTime(1500);

      expect(mockPush).toHaveBeenCalledWith('/dashboard');
    });
  });

  describe('Error Handling', () => {
    test('shows error on API failure', async () => {
      authApi.changePassword.mockRejectedValue(new Error('Current password is incorrect'));

      render(<ChangePassword />);

      fireEvent.change(screen.getByLabelText(/current password/i), { target: { value: 'wrongpassword' } });
      fireEvent.change(screen.getByLabelText(/^new password/i), { target: { value: 'NewPassword123!' } });
      fireEvent.change(screen.getByLabelText(/confirm new password/i), { target: { value: 'NewPassword123!' } });

      fireEvent.click(screen.getByRole('button', { name: /^change password$/i }));

      await waitFor(() => {
        expect(screen.getByText(/current password is incorrect/i)).toBeInTheDocument();
      });
    });

    test('shows generic error when no message provided', async () => {
      authApi.changePassword.mockRejectedValue(new Error());

      render(<ChangePassword />);

      fireEvent.change(screen.getByLabelText(/current password/i), { target: { value: 'oldpassword' } });
      fireEvent.change(screen.getByLabelText(/^new password/i), { target: { value: 'NewPassword123!' } });
      fireEvent.change(screen.getByLabelText(/confirm new password/i), { target: { value: 'NewPassword123!' } });

      fireEvent.click(screen.getByRole('button', { name: /^change password$/i }));

      await waitFor(() => {
        expect(screen.getByText(/failed to change password/i)).toBeInTheDocument();
      });
    });
  });

  describe('Form Interactions', () => {
    test('disables inputs during loading', async () => {
      authApi.changePassword.mockImplementation(() => new Promise(() => {}));

      render(<ChangePassword />);

      fireEvent.change(screen.getByLabelText(/current password/i), { target: { value: 'oldpassword' } });
      fireEvent.change(screen.getByLabelText(/^new password/i), { target: { value: 'NewPassword123!' } });
      fireEvent.change(screen.getByLabelText(/confirm new password/i), { target: { value: 'NewPassword123!' } });

      fireEvent.click(screen.getByRole('button', { name: /^change password$/i }));

      expect(screen.getByLabelText(/current password/i)).toBeDisabled();
      expect(screen.getByLabelText(/^new password/i)).toBeDisabled();
      expect(screen.getByLabelText(/confirm new password/i)).toBeDisabled();
    });

    test('disables submit button during loading', async () => {
      authApi.changePassword.mockImplementation(() => new Promise(() => {}));

      render(<ChangePassword />);

      fireEvent.change(screen.getByLabelText(/current password/i), { target: { value: 'oldpassword' } });
      fireEvent.change(screen.getByLabelText(/^new password/i), { target: { value: 'NewPassword123!' } });
      fireEvent.change(screen.getByLabelText(/confirm new password/i), { target: { value: 'NewPassword123!' } });

      fireEvent.click(screen.getByRole('button', { name: /^change password$/i }));

      expect(screen.getByRole('button', { name: /changing password/i })).toBeDisabled();
    });
  });

  describe('Logout', () => {
    test('calls logout and redirects when cancel clicked', () => {
      render(<ChangePassword />);

      fireEvent.click(screen.getByRole('button', { name: /cancel and logout/i }));

      expect(authApi.logout).toHaveBeenCalled();
      expect(mockPush).toHaveBeenCalledWith('/login');
    });
  });
});
