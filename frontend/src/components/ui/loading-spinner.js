// src/components/ui/loading-spinner.js
// Replaces scattered "Loading..." text / inline spinner patterns across pages.

import React from 'react';

/**
 * LoadingSpinner
 * @param {object} props
 * @param {string}  [props.size]    - 'sm' | 'md' | 'lg'  (default 'md')
 * @param {string}  [props.label]   - accessible text (default 'Loading…')
 * @param {boolean} [props.inline]  - render inline (default false = centered block)
 * @param {string}  [props.className]
 */
const SIZES = {
  sm: 'h-4 w-4 border-2',
  md: 'h-8 w-8 border-2',
  lg: 'h-12 w-12 border-4',
};

const LoadingSpinner = ({ size = 'md', label = 'Loading…', inline = false, className = '' }) => {
  const sizeClasses = SIZES[size] || SIZES.md;

  const spinner = (
    <span
      className={`inline-block ${sizeClasses} border-gray-300 border-t-indigo-600 rounded-full animate-spin ${className}`}
      role="status"
      aria-label={label}
    />
  );

  if (inline) return spinner;

  return (
    <div className="flex flex-col items-center justify-center py-8 gap-3 text-gray-500">
      {spinner}
      <span className="text-sm">{label}</span>
    </div>
  );
};

LoadingSpinner.displayName = 'LoadingSpinner';
export { LoadingSpinner };
export default LoadingSpinner;
