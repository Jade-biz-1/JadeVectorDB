import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Global mocks: next/head, next/router, next/link via jest.config.js
jest.mock('@/lib/api', () => require('./__mocks__/api'));
const { authApi } = require('@/lib/api');

import Register from '@/pages/register';

describe('Register page', () => {
  beforeEach(() => jest.clearAllMocks());

  it('renders the Sign Up heading', () => {
    render(<Register />);
    expect(screen.getByRole('heading', { name: /sign up/i })).toBeInTheDocument();
  });

  it('renders username, email, password, confirm-password fields', () => {
    render(<Register />);
    expect(screen.getByLabelText(/username/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/^password/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/confirm password/i)).toBeInTheDocument();
  });

  it('renders a link to login page', () => {
    render(<Register />);
    expect(screen.getByRole('link', { name: /sign in/i })).toBeInTheDocument();
  });

  it('shows error when username is too short', async () => {
    render(<Register />);
    await userEvent.type(screen.getByLabelText(/username/i), 'ab');
    await userEvent.type(screen.getByLabelText(/^password/i), 'Password1!x');
    await userEvent.type(screen.getByLabelText(/confirm password/i), 'Password1!x');
    const form = screen.getByRole('button', { name: /create account/i }).closest('form');
    fireEvent.submit(form);
    await waitFor(() => {
      // Error text is "Username must be at least 3 characters long"
      // to distinguish from the static hint "At least 3 characters"
      expect(screen.getByText(/username must be at least 3 characters/i)).toBeInTheDocument();
    });
  });

  it('shows error when passwords do not match', async () => {
    render(<Register />);
    await userEvent.type(screen.getByLabelText(/username/i), 'newuser');
    await userEvent.type(screen.getByLabelText(/^password/i), 'Password1!x');
    await userEvent.type(screen.getByLabelText(/confirm password/i), 'Different1!');
    const form = screen.getByRole('button', { name: /create account/i }).closest('form');
    fireEvent.submit(form);
    await waitFor(() => {
      expect(screen.getByText(/passwords do not match/i)).toBeInTheDocument();
    });
  });

  it('calls authApi.register on valid submission', async () => {
    authApi.register.mockResolvedValueOnce({ user_id: '123' });
    render(<Register />);

    await userEvent.type(screen.getByLabelText(/username/i), 'newuser');
    await userEvent.type(screen.getByLabelText(/^password/i), 'Password1!x');
    await userEvent.type(screen.getByLabelText(/confirm password/i), 'Password1!x');
    fireEvent.click(screen.getByRole('button', { name: /create account/i }));

    await waitFor(() => {
      expect(authApi.register).toHaveBeenCalledWith('newuser', 'Password1!x', '', []);
    });
  });

  it('shows success message after registration', async () => {
    authApi.register.mockResolvedValueOnce({ user_id: '123' });
    render(<Register />);

    await userEvent.type(screen.getByLabelText(/username/i), 'newuser');
    await userEvent.type(screen.getByLabelText(/^password/i), 'Password1!x');
    await userEvent.type(screen.getByLabelText(/confirm password/i), 'Password1!x');
    fireEvent.click(screen.getByRole('button', { name: /create account/i }));

    await waitFor(() => {
      expect(screen.getByText(/account created successfully/i)).toBeInTheDocument();
    });
  });

  it('shows error message when registration fails', async () => {
    authApi.register.mockRejectedValueOnce(new Error('Username already taken'));
    render(<Register />);

    await userEvent.type(screen.getByLabelText(/username/i), 'existinguser');
    await userEvent.type(screen.getByLabelText(/^password/i), 'Password1!x');
    await userEvent.type(screen.getByLabelText(/confirm password/i), 'Password1!x');
    fireEvent.click(screen.getByRole('button', { name: /create account/i }));

    await waitFor(() => {
      expect(screen.getByText(/username already taken/i)).toBeInTheDocument();
    });
  });

  it('disables button while loading', async () => {
    authApi.register.mockImplementation(() => new Promise(() => {}));
    render(<Register />);

    await userEvent.type(screen.getByLabelText(/username/i), 'newuser');
    await userEvent.type(screen.getByLabelText(/^password/i), 'Password1!x');
    await userEvent.type(screen.getByLabelText(/confirm password/i), 'Password1!x');
    fireEvent.click(screen.getByRole('button', { name: /create account/i }));

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /creating account/i })).toBeDisabled();
    });
  });
});
