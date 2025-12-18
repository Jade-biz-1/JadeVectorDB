import React, { useState } from 'react';

/**
 * HelpTooltip - Contextual tooltip that appears on hover
 */
const HelpTooltip = ({
  content,
  title = null,
  icon = '❓',
  position = 'top', // top, bottom, left, right
  children,
  className = ''
}) => {
  const [isVisible, setIsVisible] = useState(false);

  // Position classes
  const positionClasses = {
    top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
    left: 'right-full top-1/2 -translate-y-1/2 mr-2',
    right: 'left-full top-1/2 -translate-y-1/2 ml-2'
  };

  // Arrow position classes
  const arrowClasses = {
    top: 'top-full left-1/2 -translate-x-1/2 border-t-gray-900',
    bottom: 'bottom-full left-1/2 -translate-x-1/2 border-b-gray-900',
    left: 'left-full top-1/2 -translate-y-1/2 border-l-gray-900',
    right: 'right-full top-1/2 -translate-y-1/2 border-r-gray-900'
  };

  return (
    <div className={`relative inline-flex items-center ${className}`}>
      {/* Trigger */}
      <div
        className="relative"
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
        onFocus={() => setIsVisible(true)}
        onBlur={() => setIsVisible(false)}
      >
        {children}
      </div>

      {/* Tooltip */}
      {isVisible && (
        <div
          className={`
            absolute z-50 ${positionClasses[position]}
            bg-gray-900 text-white text-sm rounded-lg shadow-xl
            px-3 py-2 max-w-xs
            animate-fadeIn
          `}
          role="tooltip"
        >
          {/* Content */}
          <div>
            {title && (
              <div className="flex items-center gap-2 font-bold mb-1">
                <span>{icon}</span>
                <span>{title}</span>
              </div>
            )}
            <div className="text-gray-200">
              {content}
            </div>
          </div>

          {/* Arrow */}
          <div
            className={`
              absolute w-0 h-0
              border-4 border-transparent
              ${arrowClasses[position]}
            `}
          />
        </div>
      )}
    </div>
  );
};

/**
 * HelpIcon - Small help icon that shows tooltip on hover
 */
export const HelpIcon = ({ content, title, position = 'top', size = 'small' }) => {
  const sizes = {
    small: 'w-4 h-4 text-xs',
    medium: 'w-5 h-5 text-sm',
    large: 'w-6 h-6 text-base'
  };

  return (
    <HelpTooltip content={content} title={title} position={position}>
      <button
        className={`
          ${sizes[size]}
          inline-flex items-center justify-center
          rounded-full bg-blue-100 text-blue-600
          hover:bg-blue-200 hover:text-blue-700
          transition-colors cursor-help
        `}
        type="button"
        aria-label="Help"
      >
        <span className="font-bold">?</span>
      </button>
    </HelpTooltip>
  );
};

/**
 * HelpLabel - Label with help tooltip
 */
export const HelpLabel = ({ label, helpContent, helpTitle, required = false, className = '' }) => {
  return (
    <label className={`flex items-center gap-2 ${className}`}>
      <span className="font-semibold text-gray-700">
        {label}
        {required && <span className="text-red-500 ml-1">*</span>}
      </span>
      <HelpIcon content={helpContent} title={helpTitle} size="small" />
    </label>
  );
};

export default HelpTooltip;
