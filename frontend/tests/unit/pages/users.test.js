import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Global mocks: next/head, next/router, next/link via jest.config.js
jest.mock('@/components/Layout', () => {
  return ({ children }) => <div data-testid="layout">{children}</div>;
});
jest.mock('@/lib/api', () => require('./__mocks__/api'));
const { usersApi } = require('@/lib/api');

// Suppress window.confirm used in delete
global.confirm = jest.fn(() => false);

import UserManagement from '@/pages/users';

const mockUsers = [
  { user_id: 'u-1', username: 'alice', email: 'alice@example.com', roles: ['admin'], active: true },
  { user_id: 'u-2', username: 'bob', email: 'bob@example.com', roles: ['viewer'], active: false },
];

const fillAddUserForm = async (username, email, password) => {
  await userEvent.type(screen.getByPlaceholderText('john_doe'), username);
  await userEvent.type(screen.getByPlaceholderText('john@example.com'), email);
  await userEvent.type(screen.getByPlaceholderText('Enter password'), password);
};

describe('UserManagement page', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    usersApi.listUsers.mockResolvedValue({ users: mockUsers });
  });

  it('renders the page heading', () => {
    render(<UserManagement />);
    expect(screen.getByText(/user management/i)).toBeInTheDocument();
  });

  it('fetches and displays users on mount', async () => {
    render(<UserManagement />);
    await waitFor(() => {
      expect(screen.getByText('alice')).toBeInTheDocument();
      expect(screen.getByText('bob')).toBeInTheDocument();
    });
  });

  it('shows user emails in the list', async () => {
    render(<UserManagement />);
    await waitFor(() => {
      expect(screen.getByText('alice@example.com')).toBeInTheDocument();
    });
  });

  it('shows error when fetching users fails', async () => {
    usersApi.listUsers.mockRejectedValueOnce(new Error('Forbidden'));
    render(<UserManagement />);
    await waitFor(() => {
      expect(screen.getByText(/error fetching users/i)).toBeInTheDocument();
    });
  });

  it('calls createUser when add user form is submitted', async () => {
    usersApi.createUser.mockResolvedValueOnce({ user_id: 'u-3' });
    usersApi.listUsers.mockResolvedValue({ users: mockUsers });

    render(<UserManagement />);
    await waitFor(() => screen.getByText('alice'));

    await fillAddUserForm('charlie', 'charlie@example.com', 'Password1!');
    // Submit directly to bypass HTML required on email field
    const form = screen.getByPlaceholderText('john_doe').closest('form');
    fireEvent.submit(form);

    await waitFor(() => {
      expect(usersApi.createUser).toHaveBeenCalledWith(
        'charlie', 'Password1!', 'charlie@example.com', expect.any(Array)
      );
    });
  });

  it('shows success message after creating a user', async () => {
    usersApi.createUser.mockResolvedValueOnce({ user_id: 'u-3' });
    usersApi.listUsers.mockResolvedValue({ users: mockUsers });

    render(<UserManagement />);
    await waitFor(() => screen.getByText('alice'));

    await fillAddUserForm('charlie', 'charlie@example.com', 'Password1!');
    const form = screen.getByPlaceholderText('john_doe').closest('form');
    fireEvent.submit(form);

    await waitFor(() => {
      expect(screen.getByText(/user created successfully/i)).toBeInTheDocument();
    });
  });

  it('shows error when createUser fails', async () => {
    usersApi.createUser.mockRejectedValueOnce(new Error('Username already exists'));
    render(<UserManagement />);
    await waitFor(() => screen.getByText('alice'));

    await fillAddUserForm('alice', 'alice2@example.com', 'Password1!');
    const form = screen.getByPlaceholderText('john_doe').closest('form');
    fireEvent.submit(form);

    await waitFor(() => {
      expect(screen.getByText(/username already exists/i)).toBeInTheDocument();
    });
  });
});
