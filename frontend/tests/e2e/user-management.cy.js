// frontend/tests/e2e/user-management.cy.js
// E2E tests for the User Management user journey.

const USERS = [
  { user_id: 'u-1', username: 'alice', email: 'alice@example.com', roles: ['admin'],  active: true  },
  { user_id: 'u-2', username: 'bob',   email: 'bob@example.com',   roles: ['viewer'], active: false },
];

describe('User Management Journey', () => {
  beforeEach(() => {
    cy.intercept('GET', '**/api/users*', {
      statusCode: 200,
      body: { users: USERS, total: USERS.length },
    }).as('getUsers');
  });

  it('loads the user management page with the correct heading', () => {
    cy.visit('/users');
    cy.get('h1, h2').first().should('contain.text', 'User');
  });

  it('displays all users from the API', () => {
    cy.visit('/users');
    cy.wait('@getUsers');

    cy.contains('alice').should('be.visible');
    cy.contains('bob').should('be.visible');
  });

  it('shows user emails in the list', () => {
    cy.visit('/users');
    cy.wait('@getUsers');

    cy.contains('alice@example.com').should('be.visible');
  });

  it('creates a new user successfully', () => {
    cy.intercept('POST', '**/api/users*', {
      statusCode: 201,
      body: { user_id: 'u-3', username: 'charlie' },
    }).as('createUser');

    cy.visit('/users');
    cy.wait('@getUsers');

    cy.get('input[placeholder="john_doe"]').type('charlie');
    cy.get('input[placeholder="john@example.com"]').type('charlie@example.com');
    cy.get('input[placeholder="Enter password"]').type('SecurePass123!');

    cy.get('form').submit();

    cy.wait('@createUser');
    cy.contains(/user created successfully/i).should('be.visible');
  });

  it('shows an error when creating a duplicate username', () => {
    cy.intercept('POST', '**/api/users*', {
      statusCode: 409,
      body: { error: 'Username already exists' },
    }).as('createUser');

    cy.visit('/users');
    cy.wait('@getUsers');

    cy.get('input[placeholder="john_doe"]').type('alice');
    cy.get('input[placeholder="john@example.com"]').type('alice2@example.com');
    cy.get('input[placeholder="Enter password"]').type('SecurePass123!');

    cy.get('form').submit();

    cy.wait('@createUser');
    cy.contains(/username already exists/i).should('be.visible');
  });

  it('shows an error when the users API is unavailable', () => {
    cy.intercept('GET', '**/api/users*', {
      statusCode: 503,
      body: { error: 'Service unavailable' },
    }).as('getUsersFail');

    cy.visit('/users');
    cy.wait('@getUsersFail');

    cy.contains(/error fetching users/i).should('be.visible');
  });
});
