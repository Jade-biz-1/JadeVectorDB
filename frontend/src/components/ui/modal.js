// src/components/ui/modal.js
// Replaces the ~40-line modal CSS block repeated in users.js, vectors.js,
// and databases/[id].js.  Traps focus and closes on Escape / overlay-click.

import React, { useEffect } from 'react';

/**
 * Modal
 * @param {object} props
 * @param {boolean}  props.open       - controls visibility
 * @param {function} props.onClose    - called when user closes the modal
 * @param {string}   [props.title]    - heading text
 * @param {string}   [props.className] - extra classes for the dialog box
 * @param {React.ReactNode} props.children
 */
const Modal = ({ open, onClose, title, className = '', children }) => {
  // Close on Escape key
  useEffect(() => {
    if (!open) return;
    const handleKey = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
      onClick={onClose}
      aria-modal="true"
      role="dialog"
      aria-label={title}
    >
      {/* Stop click propagation so clicks inside don't close the modal */}
      <div
        className={`relative bg-white rounded-lg shadow-xl p-6 w-full max-w-lg mx-4 max-h-[90vh] overflow-y-auto ${className}`}
        onClick={(e) => e.stopPropagation()}
      >
        {title && (
          <div className="flex items-start justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900">{title}</h2>
            <button
              onClick={onClose}
              className="ml-4 text-gray-400 hover:text-gray-600 transition-colors"
              aria-label="Close modal"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        )}
        {children}
      </div>
    </div>
  );
};

Modal.displayName = 'Modal';
export { Modal };
export default Modal;
