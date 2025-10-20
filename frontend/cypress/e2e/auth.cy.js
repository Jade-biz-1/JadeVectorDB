// frontend/cypress/e2e/auth.cy.js
describe('Authentication Flow', () => {
  it('should allow user to log in', () => {
    cy.visit('/auth');

    // Check for auth page elements
    cy.get('h1').should('contain', 'Authentication & API Key Management');
    
    // Click on the Authentication tab
    cy.contains('Authentication').click();
    
    // Fill in login form
    cy.get('input[name="username"]').type('testuser');
    cy.get('input[name="password"]').type('testpassword');
    
    // Submit form
    cy.get('button').contains('Log In').click();
    
    // Check for success message or redirect
    cy.contains('Successfully logged in as testuser').should('exist');
  });

  it('should allow creating a new API key', () => {
    cy.visit('/auth');
    
    // Click on the API Keys tab
    cy.contains('API Keys').click();
    
    // Fill in API key name
    cy.get('input[placeholder="e.g., Production API Key"]').type('Test API Key');
    
    // Click create button
    cy.get('button').contains('Create API Key').click();
    
    // Check for new API key display
    cy.contains('Your new API Key').should('exist');
  });

  it('should show validation errors for mismatched passwords', () => {
    cy.visit('/auth');
    
    // Click on the Authentication tab
    cy.contains('Authentication').click();
    
    // Fill in registration form with mismatched passwords
    cy.get('input[name="newUsername"]').type('newuser');
    cy.get('input[name="newPassword"]').type('password123');
    cy.get('input[name="confirmPassword"]').type('differentpassword');
    
    // Submit registration
    cy.get('button').contains('Register').click();
    
    // Check for validation error message
    cy.contains('Passwords do not match').should('exist');
  });
});