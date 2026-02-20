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

  it('should display server info card', () => {
    // Check that the Server Info card is present with status metrics
    cy.contains('Server Info').should('exist');
    cy.contains('Uptime').should('exist');
    cy.contains('Version').should('exist');
    cy.contains('Databases').should('exist');
  });

  it('should display system resources when available', () => {
    cy.contains('System Resources').should('exist');
    cy.contains('CPU Usage').should('exist');
    cy.contains('Memory Usage').should('exist');
    cy.contains('Disk Usage').should('exist');
    cy.contains('Total Vectors').should('exist');
  });

  it('should refresh data when Refresh button is clicked', () => {
    // Click the refresh button
    cy.get('button').contains('Refresh').click();

    // Check that some databases appear (assuming there are test databases)
    cy.contains('Recent Databases').should('exist');
  });

  it('should navigate to database management page', () => {
    // Click on the databases link in the navigation
    cy.contains('Database Management').click();
    cy.url().should('include', '/databases');
  });
});
