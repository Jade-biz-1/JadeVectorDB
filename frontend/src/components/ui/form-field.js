// src/components/ui/form-field.js
// Replaces the .form-group / .form-label / .form-input pattern repeated 91
// times across 6 pages.

import React from 'react';

/**
 * FormField  — label + any input/select/textarea child
 * @param {object} props
 * @param {string}  props.label      - visible label text
 * @param {string}  [props.htmlFor]  - connects label to input id
 * @param {string}  [props.hint]     - small helper text below the field
 * @param {string}  [props.error]    - red error text below the field
 * @param {boolean} [props.required] - appends * to the label
 * @param {string}  [props.className]
 * @param {React.ReactNode} props.children
 */
const FormField = ({ label, htmlFor, hint, error, required = false, className = '', children }) => (
  <div className={`flex flex-col gap-1 ${className}`}>
    {label && (
      <label
        htmlFor={htmlFor}
        className="text-sm font-medium text-gray-700"
      >
        {label}
        {required && <span className="text-red-500 ml-1" aria-hidden="true">*</span>}
      </label>
    )}
    {children}
    {hint && !error && (
      <p className="text-xs text-gray-500">{hint}</p>
    )}
    {error && (
      <p className="text-xs text-red-600" role="alert">{error}</p>
    )}
  </div>
);

FormField.displayName = 'FormField';
export { FormField };
export default FormField;
