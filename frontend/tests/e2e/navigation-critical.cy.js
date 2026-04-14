// frontend/tests/e2e/navigation-critical.cy.js
// E2E tests for critical page navigation and page structure.
// Uses cy.intercept to prevent real API calls.

// Generic intercepts for pages that load data on mount
const stubCommonApis = () => {
  cy.intercept('GET', '**/api/databases*',    { body: { databases: [], total: 0 } }).as('dbs');
  cy.intercept('GET', '**/api/users*',        { body: { users: [],     total: 0 } }).as('users');
  cy.intercept('GET', '**/api/alerts*',       { body: { alerts: [],    total: 0 } }).as('alerts');
  cy.intercept('GET', '**/api/monitoring*',   { body: {} }).as('monitoring');
  cy.intercept('GET', '**/api/cluster*',      { body: { nodes: [] }    }).as('cluster');
  cy.intercept('GET', '**/api/security*',     { body: { logs: [] }     }).as('security');
  cy.intercept('GET', '**/api/performance*',  { body: { metrics: {} }  }).as('perf');
  cy.intercept('GET', '**/v1/**',             { body: {} }).as('v1any');
};

describe('Critical Page Navigation', () => {
  beforeEach(stubCommonApis);

  it('/ — renders the index page', () => {
    cy.visit('/');
    cy.get('body').should('exist');
  });

  it('/databases — shows Database Management heading', () => {
    cy.visit('/databases');
    cy.get('h1').should('contain.text', 'Database Management');
  });

  it('/vectors — shows Vector Management heading', () => {
    cy.visit('/vectors');
    cy.get('h1, h2').first().should('contain.text', 'Vector');
  });

  it('/search — shows Vector Search heading and form elements', () => {
    cy.visit('/search');
    cy.get('h1').should('contain.text', 'Vector Search');
    cy.get('textarea').should('exist');
    cy.get('select').should('exist');
  });

  it('/batch-operations — shows Batch Vector Operations heading', () => {
    cy.visit('/batch-operations');
    cy.get('h1, h2').first().should('contain.text', 'Batch');
  });

  it('/users — shows User Management heading', () => {
    cy.visit('/users');
    cy.get('h1, h2').first().should('contain.text', 'User');
  });

  it('/alerting — shows System Alerts heading', () => {
    cy.visit('/alerting');
    cy.get('h1').should('contain.text', 'System Alerts');
  });

  it('/cluster — shows Cluster Management heading', () => {
    cy.visit('/cluster');
    cy.get('h1, h2').first().should('contain.text', 'Cluster');
  });

  it('/security — shows Security Monitoring heading', () => {
    cy.visit('/security');
    cy.get('h1, h2').first().should('contain.text', 'Security');
  });

  it('/performance — shows Performance heading', () => {
    cy.visit('/performance');
    cy.get('h1, h2').first().should('contain.text', 'Performance');
  });

  it('/monitoring — shows System Monitoring heading', () => {
    cy.visit('/monitoring');
    cy.get('h1').should('contain.text', 'System Monitoring');
  });

  it('/login — shows Login form', () => {
    cy.visit('/login');
    cy.get('input[type="text"], input[type="email"]').should('exist');
    cy.get('input[type="password"]').should('exist');
  });

  it('/register — shows Sign Up form', () => {
    cy.visit('/register');
    cy.get('input[type="text"], input[type="email"]').should('exist');
    cy.get('input[type="password"]').should('exist');
  });

  it('/forgot-password — shows Forgot Password form', () => {
    cy.visit('/forgot-password');
    cy.get('form').should('exist');
  });
});

describe('Page Title and Meta', () => {
  beforeEach(stubCommonApis);

  it('/databases — has a title element', () => {
    cy.visit('/databases');
    cy.title().should('not.be.empty');
  });

  it('/search — has a title element', () => {
    cy.visit('/search');
    cy.title().should('not.be.empty');
  });
});
