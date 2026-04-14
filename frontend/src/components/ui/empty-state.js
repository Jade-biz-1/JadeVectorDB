// src/components/ui/empty-state.js
// Replaces the .empty-state / .empty-state-icon / .empty-state-title pattern
// repeated in databases.js, users.js, search.js, vectors.js.

import React from 'react';

/**
 * EmptyState
 * @param {object} props
 * @param {string}  [props.icon]        - emoji or short text for the large icon area
 * @param {string}  props.title         - primary message
 * @param {string}  [props.description] - secondary message
 * @param {React.ReactNode} [props.action] - optional CTA button/link
 * @param {string}  [props.className]
 */
const EmptyState = ({ icon = '📭', title, description, action, className = '' }) => (
  <div className={`flex flex-col items-center justify-center py-12 px-4 text-center ${className}`}>
    {icon && (
      <div className="text-5xl mb-4" aria-hidden="true">{icon}</div>
    )}
    {title && (
      <h3 className="text-lg font-semibold text-gray-700 mb-1">{title}</h3>
    )}
    {description && (
      <p className="text-sm text-gray-500 mb-4">{description}</p>
    )}
    {action && (
      <div className="mt-2">{action}</div>
    )}
  </div>
);

EmptyState.displayName = 'EmptyState';
export { EmptyState };
export default EmptyState;
