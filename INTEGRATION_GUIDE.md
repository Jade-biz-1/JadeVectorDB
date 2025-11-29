# Integration Guide for New Tutorial Components

This guide explains how to integrate the new components created for tasks T215.26-T215.30 into the JadeVectorDB Interactive Tutorial.

## Components Created

### 1. InteractiveAPIDocs
- Location: `/tutorials/web/src/components/InteractiveAPIDocs.jsx`
- Functionality: Interactive API documentation with runnable examples
- Features: Tabbed interface for different endpoints, language switching, copy/run functionality

### 2. BenchmarkingTools
- Location: `/tutorials/web/src/components/BenchmarkingTools.jsx`
- Functionality: Performance benchmarking tools with visualizations
- Features: Vector search, DB operations, and index operations benchmarks with charts

### 3. CommunitySharing
- Location: `/tutorials/web/src/components/CommunitySharing.jsx`
- Functionality: Community sharing features for tutorial scenarios
- Features: Share, browse, and download community-created scenarios

### 4. ResourceUsageMonitor
- Location: `/tutorials/web/src/components/ResourceUsageMonitor.jsx`
- Functionality: Monitoring resource usage for the tutorial environment
- Features: Tracks API calls, vectors stored, databases created, and memory usage

## How to Integrate Components

### Dynamic Import Method (Recommended)
To avoid SSR issues, use dynamic imports for components that use browser-specific APIs:

```javascript
import dynamic from 'next/dynamic';

const InteractiveAPIDocs = dynamic(() => import('./components/InteractiveAPIDocs'), {
  ssr: false,
  loading: () => <div>Loading API documentation...</div>
});
```

### Adding Tabs to Main Interface
To add components as tabs like in the updated version (which caused the error), ensure all components are dynamically imported:

```javascript
// In your main page component
const [activeTab, setActiveTab] = useState('tutorial'); // tutorial, api, benchmark, community, resources

return (
  <div>
    <div className="flex border-b border-gray-200">
      <button onClick={() => setActiveTab('tutorial')}>Tutorial</button>
      <button onClick={() => setActiveTab('api')}>API Docs</button>
      <button onClick={() => setActiveTab('benchmark')}>Benchmarks</button>
      <button onClick={() => setActiveTab('community')}>Community</button>
      <button onClick={() => setActiveTab('resources')}>Resources</button>
    </div>
    
    {activeTab === 'api' && <InteractiveAPIDocs />}
    {activeTab === 'benchmark' && <BenchmarkingTools />}
    {activeTab === 'community' && <CommunitySharing />}
    {activeTab === 'resources' && <ResourceUsageMonitor sessionId="tutorial-session" />}
    {activeTab === 'tutorial' && (
      // Original tutorial content
    )}
  </div>
);
```

## Backend Resource Management

The C++ resource manager includes:
- Location: `/backend/src/tutorial/resource_manager.h` and `/backend/src/tutorial/resource_manager.cpp`
- Functionality: Rate limiting and session management for the tutorial environment
- Limits: API calls, vector storage, database creation, memory usage, session timeouts

## Testing

Components can be tested using the test file:
- Location: `/frontend/src/__tests__/tutorial.test.js`
- Tests cover rendering, interactions, and functionality of all new components

## Notes

- All UI components are implemented using regular HTML elements and Tailwind CSS
- Dependencies added: chart.js and react-chartjs-2 for visualizations
- Components are designed to be modular and reusable
- Resource limits are configurable in the backend system