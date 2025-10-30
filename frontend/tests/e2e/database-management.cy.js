// frontend/tests/e2e/database-management.cy.js
describe('JadeVectorDB Database Management', () => {
  beforeEach(() => {
    cy.visit('http://localhost:3000/databases');
  });

  it('should load the database management page', () => {
    cy.title().should('include', 'Database Management');
    cy.get('h1').should('contain', 'Database Management');
  });

  it('should display the create database form', () => {
    cy.get('input[name="name"]').should('exist');
    cy.get('textarea[name="description"]').should('exist');
    cy.get('select[name="vectorDimension"]').should('exist');
    cy.get('select[name="indexType"]').should('exist');
  });

  it('should allow creating a new database', () => {
    // Fill in the database creation form
    cy.get('input[name="name"]').type('Test Database');
    cy.get('textarea[name="description"]').type('This is a test database for E2E testing');
    cy.get('select[name="vectorDimension"]').select('128');
    cy.get('select[name="indexType"]').select('FLAT');

    // Click the submit button
    cy.get('button[type="submit"]').click();

    // Verify the database was created (this would depend on your mock API)
    cy.contains('Test Database').should('be.visible');
  });

  it('should list existing databases', () => {
    // Check that databases are listed
    cy.get('[data-testid="database-list-item"]').should('have.length.greaterThan', 0);
  });
});