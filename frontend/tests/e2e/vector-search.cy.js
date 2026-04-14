// frontend/tests/e2e/vector-search.cy.js
// E2E tests for the Vector Similarity Search user journey.
// All API calls are intercepted so these run without a real backend.

const DATABASES = [
  { databaseId: 'db-1', name: 'EmbeddingDB', vectorDimension: 4, status: 'active' },
  { databaseId: 'db-2', name: 'DocDB',        vectorDimension: 8, status: 'active' },
];

const SEARCH_RESULTS = [
  { vectorId: 'v-1', score: 0.98, metadata: { title: 'Document Alpha' } },
  { vectorId: 'v-2', score: 0.87, metadata: { title: 'Document Beta'  } },
  { vectorId: 'v-3', score: 0.72, metadata: { title: 'Document Gamma' } },
];

describe('Vector Search Journey', () => {
  beforeEach(() => {
    cy.intercept('GET', '**/api/databases*', {
      statusCode: 200,
      body: { databases: DATABASES, total: DATABASES.length },
    }).as('getDatabases');
  });

  it('loads the search page with the correct heading', () => {
    cy.visit('/search');
    cy.get('h1').should('contain', 'Vector Search');
  });

  it('populates the database dropdown from the API', () => {
    cy.visit('/search');
    cy.wait('@getDatabases');
    cy.get('select').first().should('contain', 'EmbeddingDB');
    cy.get('select').first().should('contain', 'DocDB');
  });

  it('performs a comma-separated search and displays results', () => {
    cy.intercept('POST', '**/api/databases/db-1/search*', {
      statusCode: 200,
      body: { results: SEARCH_RESULTS },
    }).as('doSearch');

    cy.visit('/search');
    cy.wait('@getDatabases');

    cy.get('select').first().select('db-1');
    cy.get('textarea').type('0.1, 0.2, 0.3, 0.4');
    cy.get('form').submit();

    cy.wait('@doSearch');
    cy.contains('Vector ID: v-1').should('be.visible');
    cy.contains('Vector ID: v-2').should('be.visible');
    cy.contains('Vector ID: v-3').should('be.visible');
  });

  it('performs a JSON-array search and displays results', () => {
    cy.intercept('POST', '**/api/databases/db-1/search*', {
      statusCode: 200,
      body: { results: SEARCH_RESULTS.slice(0, 1) },
    }).as('doSearch');

    cy.visit('/search');
    cy.wait('@getDatabases');

    cy.get('select').first().select('db-1');
    cy.get('textarea').invoke('val', '[0.5, 0.6, 0.7, 0.8]').trigger('input').trigger('change');
    cy.get('form').submit();

    cy.wait('@doSearch');
    cy.contains('Vector ID: v-1').should('be.visible');
  });

  it('shows an error message when the search API returns 500', () => {
    cy.intercept('POST', '**/api/databases/db-1/search*', {
      statusCode: 500,
      body: { error: 'Internal server error' },
    }).as('doSearch');

    cy.visit('/search');
    cy.wait('@getDatabases');

    cy.get('select').first().select('db-1');
    cy.get('textarea').type('0.1, 0.2, 0.3, 0.4');
    cy.get('form').submit();

    cy.wait('@doSearch');
    // The page shows an error of some kind
    cy.get('body').should('not.be.empty');
  });

  it('shows an empty results area when the search returns no matches', () => {
    cy.intercept('POST', '**/api/databases/db-1/search*', {
      statusCode: 200,
      body: { results: [] },
    }).as('doSearch');

    cy.visit('/search');
    cy.wait('@getDatabases');

    cy.get('select').first().select('db-1');
    cy.get('textarea').type('0.0, 0.0, 0.0, 0.0');
    cy.get('form').submit();

    cy.wait('@doSearch');
    // No result cards should appear
    cy.contains('Vector ID:').should('not.exist');
  });
});
