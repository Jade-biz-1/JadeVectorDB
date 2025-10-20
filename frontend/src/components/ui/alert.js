// frontend/src/components/ui/alert.js

import React from 'react';

const Alert = React.forwardRef(({ className, variant = 'default', ...props }, ref) => {
  const variantClasses = {
    default: 'border border-gray-200 bg-white text-foreground',
    destructive: 'border-red-200/50 text-destructive dark:border-red-900 [&>svg]:text-destructive'
  };
  
  return (
    <div
      ref={ref}
      role="alert"
      className={`relative w-full rounded-lg border p-4 ${variantClasses[variant]} ${className || ''}`}
      {...props}
    />
  );
});
Alert.displayName = "Alert";

const AlertTitle = React.forwardRef(({ className, ...props }, ref) => (
  <h5
    ref={ref}
    className={`mb-1 font-medium leading-none tracking-tight ${className || ''}`}
    {...props}
  />
));
AlertTitle.displayName = "AlertTitle";

const AlertDescription = React.forwardRef(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={`text-sm [&_p]:leading-relaxed ${className || ''}`}
    {...props}
  />
));
AlertDescription.displayName = "AlertDescription";

export { 
  Alert,
  AlertTitle,
  AlertDescription 
};