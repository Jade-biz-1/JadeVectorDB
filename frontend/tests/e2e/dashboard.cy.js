// frontend/tests/e2e/dashboard.cy.js
describe('JadeVectorDB Dashboard', () => {
  beforeEach(() => {
    // Set up any required state before each test
    cy.visit('http://localhost:3000');
  });

  it('should load the dashboard with correct title', () => {
    cy.title().should('include', 'JadeVectorDB Dashboard');
    cy.get('h1').should('contain', 'JadeVectorDB Dashboard');
  });

  it('should display system overview metrics', () => {
    // Check that the metrics cards are present
    cy.get('[data-testid="total-databases"]').should('exist');
    cy.get('[data-testid="total-vectors"]').should('exist');
    cy.get('[data-testid="total-indexes"]').should('exist');
  });

  it('should refresh databases when Refresh button is clicked', () => {
    // Click the refresh button
    cy.get('button').contains('Refresh Databases').click();
    
    // Check that some databases appear (assuming there are test databases)
    cy.get('[data-testid="database-card"]').should('have.length.greaterThan', 0);
  });

  it('should navigate to database management page', () => {
    // Click on the databases link in the navigation
    cy.contains('Database Management').click();
    cy.url().should('include', '/databases');
  });
});