import React from 'react';
import dynamic from 'next/dynamic';

// Dynamically import LivePreviewPanel with SSR disabled
const LivePreviewPanel = dynamic(() => import('./LivePreviewPanelImpl'), {
  ssr: false,
  loading: () => <div className="module-card">Loading preview...</div>
});

export default LivePreviewPanel;