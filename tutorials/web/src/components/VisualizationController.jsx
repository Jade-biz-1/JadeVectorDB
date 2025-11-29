import React, { useState } from 'react';
import dynamic from 'next/dynamic';

// Dynamically import components with SSR disabled
const EnhancedVisualDashboard = dynamic(() => import('./EnhancedVisualDashboard'), {
  ssr: false,
  loading: () => <div className="border border-gray-200 rounded-lg p-4">Loading visualization...</div>
});

const ThreeDVisualization = dynamic(() => import('./ThreeDVisualization'), {
  ssr: false,
  loading: () => <div className="border border-gray-200 rounded-lg p-4 h-[500px]">Loading 3D visualization...</div>
});

const VisualizationController = () => {
  const [viewType, setViewType] = useState('2d'); // '2d' or '3d'
  const [visualizationType, setVisualizationType] = useState('tsne'); // 'tsne', 'pca', 'umap'
  const [similarityMetric, setSimilarityMetric] = useState('cosine'); // 'cosine', 'euclidean', 'dot'

  return (
    <div className="module-card">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-4">
        <h2 className="text-xl font-semibold text-gray-800 mb-2 md:mb-0">Vector Space Visualization</h2>
        
        <div className="flex flex-wrap gap-3">
          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-700">View:</span>
            <div className="flex bg-gray-100 rounded-lg p-1">
              <button
                className={`px-3 py-1 text-sm rounded-md transition-colors ${
                  viewType === '2d' 
                    ? 'bg-white text-gray-800 shadow-sm' 
                    : 'text-gray-600 hover:text-gray-800'
                }`}
                onClick={() => setViewType('2d')}
              >
                2D
              </button>
              <button
                className={`px-3 py-1 text-sm rounded-md transition-colors ${
                  viewType === '3d' 
                    ? 'bg-white text-gray-800 shadow-sm' 
                    : 'text-gray-600 hover:text-gray-800'
                }`}
                onClick={() => setViewType('3d')}
              >
                3D
              </button>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-700">Type:</span>
            <select 
              value={visualizationType} 
              onChange={(e) => setVisualizationType(e.target.value)}
              className="p-1 border border-gray-300 rounded-md text-sm"
            >
              <option value="tsne">t-SNE</option>
              <option value="pca">PCA</option>
              <option value="umap">UMAP</option>
            </select>
          </div>
          
          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-700">Metric:</span>
            <select 
              value={similarityMetric} 
              onChange={(e) => setSimilarityMetric(e.target.value)}
              className="p-1 border border-gray-300 rounded-md text-sm"
            >
              <option value="cosine">Cosine</option>
              <option value="euclidean">Euclidean</option>
              <option value="dot">Dot Product</option>
            </select>
          </div>
        </div>
      </div>
      
      <div className="border border-gray-200 rounded-lg bg-white overflow-hidden">
        {viewType === '2d' ? (
          <EnhancedVisualDashboard />
        ) : (
          <div className="relative h-[500px] w-full">
            <ThreeDVisualization width="100%" height="500" />
          </div>
        )}
      </div>
      
      <div className="mt-4 grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-blue-50 p-3 rounded-lg border border-blue-100">
          <div className="text-lg font-bold text-blue-800">10,000</div>
          <div className="text-sm text-blue-700">Total Vectors</div>
        </div>
        <div className="bg-green-50 p-3 rounded-lg border border-green-100">
          <div className="text-lg font-bold text-green-800">0.87</div>
          <div className="text-sm text-green-700">Avg Similarity</div>
        </div>
        <div className="bg-purple-50 p-3 rounded-lg border border-purple-100">
          <div className="text-lg font-bold text-purple-800">768</div>
          <div className="text-sm text-purple-700">Dimensions</div>
        </div>
        <div className="bg-orange-50 p-3 rounded-lg border border-orange-100">
          <div className="text-lg font-bold text-orange-800">5</div>
          <div className="text-sm text-orange-700">Clusters</div>
        </div>
      </div>
      
      <div className="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
        <h3 className="font-medium text-gray-800 mb-2">
          {viewType === '2d' ? '2D Visualization' : '3D Visualization'} Information
        </h3>
        <p className="text-sm text-gray-600">
          {viewType === '2d' 
            ? 'This 2D projection uses t-SNE to visualize high-dimensional vector relationships. Red represents the query vector, blue vectors are similar items, purple clusters show groups of related vectors, and orange represents background vectors.'
            : 'This 3D visualization shows vectors in three-dimensional space. Rotate by clicking and dragging, zoom with scroll wheel. Red is the query vector, blue vectors are similar items, and different colored clusters show groups of related vectors.'}
        </p>
      </div>
    </div>
  );
};

export default VisualizationController;