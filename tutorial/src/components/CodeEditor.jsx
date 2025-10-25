import React from 'react';
import dynamic from 'next/dynamic';

// Dynamically import AdvancedCodeEditor with SSR disabled
const AdvancedCodeEditor = dynamic(() => import('./AdvancedCodeEditor'), {
  ssr: false,
  loading: () => <div className="border border-gray-200 rounded-lg p-4">Loading code editor...</div>
});

const CodeEditor = () => {
  return (
    <AdvancedCodeEditor />
  );
};

export default CodeEditor;