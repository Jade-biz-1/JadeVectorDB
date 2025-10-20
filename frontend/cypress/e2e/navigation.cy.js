// frontend/cypress/e2e/navigation.cy.js
describe('Navigation between pages', () => {
  before(() => {
    // Set up any necessary data before running tests
    cy.visit('/');
  });

  it('should navigate to the databases page', () => {
    // Visit the main page
    cy.visit('/');

    // Click on a navigation element or link that should take us to the databases page
    // Since we don't have a standard navigation bar, we'll test by visiting directly
    cy.visit('/databases');
    
    // Check that we're on the databases page by looking for specific content
    cy.get('h1').should('contain', 'Database Management');
    
    // Check for the presence of key elements
    cy.get('form').should('exist');
    cy.get('select').should('exist');
    cy.get('input').should('exist');
  });

  it('should navigate to the search page', () => {
    cy.visit('/search');
    
    cy.get('h1').should('contain', 'Vector Search');
    
    // Check for search form elements
    cy.get('select').should('exist');  // Database selection
    cy.get('textarea').should('exist'); // Query vector input
    cy.get('input[type="number"]').should('exist'); // Top K input
  });

  it('should navigate to the monitoring page', () => {
    cy.visit('/monitoring');
    
    cy.get('h1').should('contain', 'System Monitoring');
    
    // Check for monitoring page elements
    cy.get('.bg-green-600').should('exist'); // Status indicator
    cy.get('ul').should('exist'); // Database list
  });

  it('should navigate to the indexes page', () => {
    cy.visit('/indexes');
    
    cy.get('h1').should('contain', 'Index Management');
    
    // Check for index management elements
    cy.get('select').should('exist'); // Database selection
    cy.get('form').should('exist'); // Create index form
  });

  it('should navigate to the embeddings page', () => {
    cy.visit('/embeddings');
    
    cy.get('h1').should('contain', 'Embedding Generation');
    
    // Check for embedding generation elements
    cy.get('textarea').should('exist'); // Input text
    cy.get('select').should('exist'); // Model selection
    cy.get('button').contains('Generate Embedding').should('exist');
  });
});