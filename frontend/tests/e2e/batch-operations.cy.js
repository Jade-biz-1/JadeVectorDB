// frontend/tests/e2e/batch-operations.cy.js
// E2E tests for the Batch Vector Operations user journey.

const DATABASES = [
  { databaseId: 'db-1', name: 'EmbeddingDB', vectorDimension: 4, status: 'active' },
];

describe('Batch Vector Operations Journey', () => {
  beforeEach(() => {
    cy.intercept('GET', '**/api/databases*', {
      statusCode: 200,
      body: { databases: DATABASES, total: 1 },
    }).as('getDatabases');
  });

  it('loads the batch operations page with the correct heading', () => {
    cy.visit('/batch-operations');
    cy.get('h1, h2').first().should('contain.text', 'Batch');
  });

  it('shows the database selector and Upload Vectors button by default', () => {
    cy.visit('/batch-operations');
    cy.wait('@getDatabases');

    cy.get('select').should('exist');
    cy.contains('button', /upload vectors/i).should('exist');
  });

  it('populates the database dropdown', () => {
    cy.visit('/batch-operations');
    cy.wait('@getDatabases');

    cy.get('select').should('contain', 'EmbeddingDB');
  });

  it('switches to Download mode via the radio button', () => {
    cy.visit('/batch-operations');
    cy.wait('@getDatabases');

    cy.contains('label', /download/i).click();
    cy.contains('button', /download vectors/i).should('exist');
  });

  it('adds a second vector row when Add Vector is clicked', () => {
    cy.visit('/batch-operations');
    cy.wait('@getDatabases');

    cy.contains('button', /add vector/i).click();
    cy.contains('button', /remove/i).should('have.length', 2);
  });

  it('uploads vectors to the selected database', () => {
    cy.intercept('POST', '**/api/databases/db-1/vectors/batch*', {
      statusCode: 200,
      body: { count: 1, stored: 1 },
    }).as('batchUpload');

    cy.visit('/batch-operations');
    cy.wait('@getDatabases');

    cy.get('select').select('db-1');
    cy.get('input[placeholder="Vector ID"]').type('e2e-vec-1');
    cy.get('input[placeholder="Comma-separated or JSON array"]').type('0.1, 0.2, 0.3, 0.4');
    cy.contains('button', /upload vectors/i).click();

    cy.wait('@batchUpload').its('request.body').should('be.an', 'array');
  });

  it('removes a vector row when Remove is clicked', () => {
    cy.visit('/batch-operations');
    cy.wait('@getDatabases');

    // Add a row first
    cy.contains('button', /add vector/i).click();
    cy.contains('button', /remove/i).should('have.length', 2);

    // Remove one
    cy.contains('button', /remove/i).first().click();
    cy.contains('button', /remove/i).should('have.length', 1);
  });
});
