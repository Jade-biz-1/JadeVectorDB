// frontend/tests/e2e/alerting.cy.js
// E2E tests for the Alerting / System Alerts user journey.

const ALERTS = [
  { id: 'a-1', type: 'error',   message: 'Disk space critical', timestamp: '2026-04-13T10:00:00Z' },
  { id: 'a-2', type: 'warning', message: 'Memory high',         timestamp: '2026-04-13T10:01:00Z' },
  { id: 'a-3', type: 'info',    message: 'Backup completed',    timestamp: '2026-04-13T10:02:00Z' },
];

describe('Alerting Journey', () => {
  beforeEach(() => {
    cy.intercept('GET', '**/api/alerts*', {
      statusCode: 200,
      body: { alerts: ALERTS, total: ALERTS.length },
    }).as('getAlerts');
  });

  it('loads the alerting page with the correct heading', () => {
    cy.visit('/alerting');
    cy.get('h1').should('contain.text', 'System Alerts');
  });

  it('displays all alerts on mount', () => {
    cy.visit('/alerting');
    cy.wait('@getAlerts');

    cy.contains('Disk space critical').should('be.visible');
    cy.contains('Memory high').should('be.visible');
    cy.contains('Backup completed').should('be.visible');
  });

  it('shows the alert count in the heading', () => {
    cy.visit('/alerting');
    cy.wait('@getAlerts');

    cy.contains('(3)').should('be.visible');
  });

  it('filters to show only error alerts', () => {
    cy.visit('/alerting');
    cy.wait('@getAlerts');

    cy.get('select').select('error');

    cy.contains('Disk space critical').should('be.visible');
    cy.contains('Memory high').should('not.exist');
    cy.contains('Backup completed').should('not.exist');
  });

  it('shows all alerts again when filter is reset to "all"', () => {
    cy.visit('/alerting');
    cy.wait('@getAlerts');

    cy.get('select').select('error');
    cy.get('select').select('all');

    cy.contains('Disk space critical').should('be.visible');
    cy.contains('Memory high').should('be.visible');
    cy.contains('Backup completed').should('be.visible');
  });

  it('shows zero count when no alerts exist', () => {
    cy.intercept('GET', '**/api/alerts*', {
      statusCode: 200,
      body: { alerts: [], total: 0 },
    }).as('getEmptyAlerts');

    cy.visit('/alerting');
    cy.wait('@getEmptyAlerts');

    cy.contains('(0)').should('be.visible');
  });
});
