// src/components/ui/status-badge.js
// Replaces the .badge / .badge-success / .badge-error / .badge-warning pattern
// defined independently in 6 pages (dashboard, databases, users, analytics,
// monitoring, alerting).

import React from 'react';

const STATUS_CLASSES = {
  // semantic statuses
  active:    'bg-green-100 text-green-800',
  inactive:  'bg-gray-100  text-gray-600',
  success:   'bg-green-100 text-green-800',
  failure:   'bg-red-100   text-red-800',
  error:     'bg-red-100   text-red-800',
  warning:   'bg-yellow-100 text-yellow-800',
  info:      'bg-blue-100  text-blue-800',
  online:    'bg-green-100 text-green-800',
  offline:   'bg-red-100   text-red-800',
  degraded:  'bg-yellow-100 text-yellow-800',
  pending:   'bg-yellow-100 text-yellow-800',
  // fallback
  default:   'bg-gray-100  text-gray-700',
};

/**
 * StatusBadge
 * @param {object} props
 * @param {string} props.status     - one of the keys above, or any string
 * @param {string} [props.label]    - override display text (defaults to status)
 * @param {string} [props.className]
 */
const StatusBadge = ({ status = 'default', label, className = '' }) => {
  const key = (status || 'default').toLowerCase();
  const colours = STATUS_CLASSES[key] || STATUS_CLASSES.default;

  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold ${colours} ${className}`}
    >
      {label ?? status}
    </span>
  );
};

StatusBadge.displayName = 'StatusBadge';
export { StatusBadge };
export default StatusBadge;
