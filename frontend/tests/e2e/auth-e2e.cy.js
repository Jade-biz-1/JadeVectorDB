// frontend/tests/e2e/auth-e2e.cy.js
// Cypress end-to-end tests for authentication workflows

describe('Authentication E2E Tests', () => {
  beforeEach(() => {
    // Clear localStorage before each test
    cy.clearLocalStorage();

    // Visit the auth page
    cy.visit('/auth');
  });

  // ============================================================================
  // Login Workflow Tests
  // ============================================================================

  describe('Login Workflow', () => {
    it('successfully logs in with valid credentials', () => {
      // Intercept login API call
      cy.intercept('POST', '**/v1/auth/login', {
        statusCode: 200,
        body: {
          token: 'test-jwt-token-12345',
          user: {
            id: 'user-123',
            username: 'testuser'
          }
        }
      }).as('loginRequest');

      // Intercept API keys list call
      cy.intercept('GET', '**/v1/apikeys*', {
        statusCode: 200,
        body: {
          apiKeys: [],
          total: 0
        }
      }).as('listApiKeys');

      // Click on Authentication tab
      cy.contains('Authentication').click();

      // Fill in login form
      cy.get('input[name="username"]').type('testuser');
      cy.get('input[name="password"]').type('securePassword123');

      // Submit login
      cy.contains('button', 'Log In').click();

      // Wait for API call
      cy.wait('@loginRequest');

      // Check that localStorage was updated
      cy.window().then((win) => {
        expect(win.localStorage.getItem('jadevectordb_authenticated')).to.equal('true');
        expect(win.localStorage.getItem('jadevectordb_api_key')).to.equal('test-jwt-token-12345');
        expect(win.localStorage.getItem('jadevectordb_username')).to.equal('testuser');
      });
    });

    it('shows error message for invalid credentials', () => {
      cy.intercept('POST', '**/v1/auth/login', {
        statusCode: 401,
        body: {
          message: 'Invalid username or password'
        }
      }).as('loginRequest');

      cy.contains('Authentication').click();

      cy.get('input[name="username"]').type('wronguser');
      cy.get('input[name="password"]').type('wrongpassword');

      cy.contains('button', 'Log In').click();

      cy.wait('@loginRequest');

      // Check for error alert
      cy.on('window:alert', (text) => {
        expect(text).to.contains('Login failed');
      });
    });

    it('disables login button during submission', () => {
      cy.intercept('POST', '**/v1/auth/login', (req) => {
        // Delay response to simulate slow network
        req.reply((res) => {
          setTimeout(() => {
            res.send({
              statusCode: 200,
              body: { token: 'test-token' }
            });
          }, 1000);
        });
      }).as('loginRequest');

      cy.contains('Authentication').click();

      cy.get('input[name="username"]').type('testuser');
      cy.get('input[name="password"]').type('password');

      const loginButton = cy.contains('button', 'Log In');
      loginButton.click();

      // Button should be disabled during loading
      loginButton.should('be.disabled');
    });

    it('validates required fields', () => {
      cy.contains('Authentication').click();

      // Try to submit without filling fields
      cy.contains('button', 'Log In').click();

      // Form should have validation (HTML5 required attribute)
      cy.get('input[name="username"]').should('have.attr', 'required');
      cy.get('input[name="password"]').should('have.attr', 'required');
    });
  });

  // ============================================================================
  // Registration Workflow Tests
  // ============================================================================

  describe('Registration Workflow', () => {
    it('successfully registers a new user', () => {
      cy.intercept('POST', '**/v1/auth/register', {
        statusCode: 201,
        body: {
          userId: 'new-user-id',
          username: 'newuser',
          message: 'Registration successful'
        }
      }).as('registerRequest');

      cy.contains('Authentication').click();

      // Fill registration form
      cy.get('input[name="username"]').type('newuser');
      cy.get('input[name="password"]').type('SecurePassword123!');
      cy.get('input[name="confirmPassword"]').type('SecurePassword123!');

      // Click register button
      cy.contains('button', 'Register').click();

      cy.wait('@registerRequest');

      // Check for success alert
      cy.on('window:alert', (text) => {
        expect(text).to.contains('registered successfully');
      });
    });

    it('shows error when passwords do not match', () => {
      cy.contains('Authentication').click();

      cy.get('input[name="username"]').type('newuser');
      cy.get('input[name="password"]').type('Password123!');
      cy.get('input[name="confirmPassword"]').type('DifferentPassword123!');

      cy.contains('button', 'Register').click();

      // Check for password mismatch alert
      cy.on('window:alert', (text) => {
        expect(text).to.contains('Passwords do not match');
      });
    });

    it('shows error for duplicate username', () => {
      cy.intercept('POST', '**/v1/auth/register', {
        statusCode: 409,
        body: {
          message: 'Username already exists'
        }
      }).as('registerRequest');

      cy.contains('Authentication').click();

      cy.get('input[name="username"]').type('existinguser');
      cy.get('input[name="password"]').type('Password123!');
      cy.get('input[name="confirmPassword"]').type('Password123!');

      cy.contains('button', 'Register').click();

      cy.wait('@registerRequest');

      cy.on('window:alert', (text) => {
        expect(text).to.contains('Username already exists');
      });
    });

    it('clears form fields after successful registration', () => {
      cy.intercept('POST', '**/v1/auth/register', {
        statusCode: 201,
        body: { userId: 'new-id', username: 'newuser' }
      }).as('registerRequest');

      cy.contains('Authentication').click();

      cy.get('input[name="username"]').type('newuser');
      cy.get('input[name="password"]').type('Password123!');
      cy.get('input[name="confirmPassword"]').type('Password123!');

      cy.contains('button', 'Register').click();

      cy.wait('@registerRequest');

      // Form should be cleared
      cy.get('input[name="username"]').should('have.value', '');
      cy.get('input[name="password"]').should('have.value', '');
      cy.get('input[name="confirmPassword"]').should('have.value', '');
    });
  });

  // ============================================================================
  // Logout Workflow Tests
  // ============================================================================

  describe('Logout Workflow', () => {
    beforeEach(() => {
      // Set up authenticated state
      cy.window().then((win) => {
        win.localStorage.setItem('jadevectordb_authenticated', 'true');
        win.localStorage.setItem('jadevectordb_api_key', 'test-token-123');
        win.localStorage.setItem('jadevectordb_username', 'testuser');
      });

      // Intercept API keys list
      cy.intercept('GET', '**/v1/apikeys*', {
        statusCode: 200,
        body: { apiKeys: [] }
      }).as('listApiKeys');

      cy.reload();
    });

    it('successfully logs out user', () => {
      cy.intercept('POST', '**/v1/auth/logout', {
        statusCode: 200,
        body: { success: true }
      }).as('logoutRequest');

      cy.contains('Authentication').click();

      // Find and click logout button
      cy.contains('button', 'Log Out').click();

      cy.wait('@logoutRequest');

      // Check localStorage was cleared
      cy.window().then((win) => {
        expect(win.localStorage.getItem('jadevectordb_authenticated')).to.be.null;
        expect(win.localStorage.getItem('jadevectordb_api_key')).to.be.null;
        expect(win.localStorage.getItem('jadevectordb_username')).to.be.null;
      });

      // Should show login form again
      cy.contains('button', 'Log In').should('be.visible');
    });

    it('clears localStorage even if API call fails', () => {
      cy.intercept('POST', '**/v1/auth/logout', {
        statusCode: 500,
        body: { message: 'Server error' }
      }).as('logoutRequest');

      cy.contains('Authentication').click();

      cy.contains('button', 'Log Out').click();

      // Wait a bit for the operation
      cy.wait(500);

      // Should still clear localStorage
      cy.window().then((win) => {
        expect(win.localStorage.getItem('jadevectordb_authenticated')).to.be.null;
      });
    });
  });

  // ============================================================================
  // API Key Management Workflow Tests
  // ============================================================================

  describe('API Key Management', () => {
    beforeEach(() => {
      // Set up authenticated state
      cy.window().then((win) => {
        win.localStorage.setItem('jadevectordb_authenticated', 'true');
        win.localStorage.setItem('jadevectordb_api_key', 'admin-token');
      });

      cy.intercept('GET', '**/v1/apikeys*', {
        statusCode: 200,
        body: {
          apiKeys: [
            {
              keyId: 'key-1',
              name: 'Existing Key',
              createdAt: '2025-01-01T00:00:00Z',
              permissions: ['read']
            }
          ]
        }
      }).as('listApiKeys');

      cy.reload();
    });

    it('displays list of existing API keys', () => {
      // Wait for API keys to load
      cy.wait('@listApiKeys');

      // Should show the existing key
      cy.contains('Existing Key').should('be.visible');
    });

    it('creates a new API key', () => {
      const newApiKey = 'sk_test_1234567890abcdef';

      cy.intercept('POST', '**/v1/apikeys', {
        statusCode: 201,
        body: {
          apiKey: newApiKey,
          keyId: 'key-2',
          name: 'New Test Key',
          permissions: ['read', 'write']
        }
      }).as('createApiKey');

      cy.intercept('GET', '**/v1/apikeys*', {
        statusCode: 200,
        body: {
          apiKeys: [
            {
              keyId: 'key-1',
              name: 'Existing Key',
              createdAt: '2025-01-01T00:00:00Z',
              permissions: ['read']
            },
            {
              keyId: 'key-2',
              name: 'New Test Key',
              createdAt: '2025-01-02T00:00:00Z',
              permissions: ['read', 'write']
            }
          ]
        }
      }).as('refreshApiKeys');

      // Fill in API key name
      cy.get('input[name="keyName"]').type('New Test Key');

      // Click create button
      cy.contains('button', 'Create API Key').click();

      cy.wait('@createApiKey');
      cy.wait('@refreshApiKeys');

      // Should display the generated API key
      cy.contains('Your new API Key').should('be.visible');
      cy.contains(newApiKey).should('be.visible');
    });

    it('shows error when creating key without name', () => {
      // Try to create without name
      cy.contains('button', 'Create API Key').click();

      // Should show error alert
      cy.on('window:alert', (text) => {
        expect(text).to.contains('Please enter a name');
      });
    });

    it('copies API key to clipboard', () => {
      const newApiKey = 'sk_test_copytest123';

      cy.intercept('POST', '**/v1/apikeys', {
        statusCode: 201,
        body: {
          apiKey: newApiKey,
          keyId: 'key-copy',
          name: 'Copy Test Key'
        }
      }).as('createApiKey');

      cy.get('input[name="keyName"]').type('Copy Test Key');
      cy.contains('button', 'Create API Key').click();

      cy.wait('@createApiKey');

      // Click copy button
      cy.contains('button', 'Copy').click();

      // Check clipboard (requires clipboard permissions)
      cy.window().then((win) => {
        win.navigator.clipboard.readText().then((text) => {
          expect(text).to.equal(newApiKey);
        });
      });
    });
  });

  // ============================================================================
  // Tab Navigation Tests
  // ============================================================================

  describe('Tab Navigation', () => {
    it('switches between Authentication and API Keys tabs', () => {
      // Should start on API Keys tab
      cy.contains('Manage API keys for programmatic access').should('be.visible');

      // Click Authentication tab
      cy.contains('Authentication').click();
      cy.contains('Log in to access the system').should('be.visible');

      // Click back to API Keys tab
      cy.contains('API Keys').click();
      cy.contains('Manage API keys for programmatic access').should('be.visible');
    });

    it('preserves form state when switching tabs', () => {
      cy.contains('Authentication').click();

      // Fill in username
      cy.get('input[name="username"]').type('testuser');

      // Switch to API Keys tab
      cy.contains('API Keys').click();

      // Switch back to Authentication
      cy.contains('Authentication').click();

      // Username should still be there
      cy.get('input[name="username"]').should('have.value', 'testuser');
    });
  });

  // ============================================================================
  // Session Persistence Tests
  // ============================================================================

  describe('Session Persistence', () => {
    it('maintains authentication state across page reloads', () => {
      // Log in first
      cy.intercept('POST', '**/v1/auth/login', {
        statusCode: 200,
        body: {
          token: 'persistent-token',
          user: { id: 'user-1', username: 'persistentuser' }
        }
      }).as('loginRequest');

      cy.intercept('GET', '**/v1/apikeys*', {
        statusCode: 200,
        body: { apiKeys: [] }
      }).as('listApiKeys');

      cy.contains('Authentication').click();

      cy.get('input[name="username"]').type('persistentuser');
      cy.get('input[name="password"]').type('password123');
      cy.contains('button', 'Log In').click();

      cy.wait('@loginRequest');

      // Reload the page
      cy.reload();

      // Should still be authenticated
      cy.window().then((win) => {
        expect(win.localStorage.getItem('jadevectordb_authenticated')).to.equal('true');
      });

      // Should fetch API keys automatically
      cy.wait('@listApiKeys');
    });

    it('requires re-login after logout', () => {
      // Set up authenticated state
      cy.window().then((win) => {
        win.localStorage.setItem('jadevectordb_authenticated', 'true');
        win.localStorage.setItem('jadevectordb_api_key', 'test-token');
      });

      cy.intercept('GET', '**/v1/apikeys*', {
        statusCode: 200,
        body: { apiKeys: [] }
      });

      cy.intercept('POST', '**/v1/auth/logout', {
        statusCode: 200,
        body: { success: true }
      }).as('logoutRequest');

      cy.reload();

      cy.contains('Authentication').click();
      cy.contains('button', 'Log Out').click();

      cy.wait('@logoutRequest');

      // Reload page
      cy.reload();

      // Should not be authenticated
      cy.contains('Authentication').click();
      cy.contains('button', 'Log In').should('be.visible');
    });
  });

  // ============================================================================
  // Error Handling Tests
  // ============================================================================

  describe('Error Handling', () => {
    it('handles network errors gracefully', () => {
      cy.intercept('POST', '**/v1/auth/login', {
        forceNetworkError: true
      }).as('loginRequest');

      cy.contains('Authentication').click();

      cy.get('input[name="username"]').type('testuser');
      cy.get('input[name="password"]').type('password');
      cy.contains('button', 'Log In').click();

      // Should show error message
      cy.on('window:alert', (text) => {
        expect(text).to.contains('failed');
      });
    });

    it('handles server errors', () => {
      cy.intercept('POST', '**/v1/auth/login', {
        statusCode: 500,
        body: {
          message: 'Internal server error'
        }
      }).as('loginRequest');

      cy.contains('Authentication').click();

      cy.get('input[name="username"]').type('testuser');
      cy.get('input[name="password"]').type('password');
      cy.contains('button', 'Log In').click();

      cy.wait('@loginRequest');

      cy.on('window:alert', (text) => {
        expect(text).to.contains('Internal server error');
      });
    });
  });
});
