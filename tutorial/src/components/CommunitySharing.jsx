import React, { useState } from 'react';

const CommunitySharing = () => {
  const [activeTab, setActiveTab] = useState('share');
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [code, setCode] = useState('');
  const [tags, setTags] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [copiedId, setCopiedId] = useState(null);
  const [showCode, setShowCode] = useState({});

  // Sample shared scenarios from community
  const [communityScenarios] = useState([
    {
      id: '1',
      title: 'Product Similarity Search',
      description: 'A tutorial scenario that demonstrates product similarity search using vector embeddings',
      author: 'vector_ninja',
      likes: 42,
      downloads: 128,
      tags: ['search', 'e-commerce', 'similarity'],
      code: `// Create a product database with 128-dimensional vectors
const db = await createDatabase({
  name: 'product_embeddings',
  dimensions: 128
});

// Add sample products
const products = [
  { id: 'p1', vector: [0.1, 0.5, 0.9, /* ... */], metadata: { category: 'shoes', price: 99.99 } },
  { id: 'p2', vector: [0.8, 0.3, 0.2, /* ... */], metadata: { category: 'shoes', price: 79.99 } },
  // ...
];

for (const product of products) {
  await addVector(db.id, product);
}

// Perform similarity search
const queryVector = [0.75, 0.45, 0.35, /* ... */];
const results = await similaritySearch(db.id, queryVector, { top_k: 5 });`,
      createdAt: '2023-10-10',
      stars: 4.7
    },
    {
      id: '2',
      title: 'Document Semantic Search',
      description: 'Semantic search for documents using vector embeddings',
      author: 'doc_reader',
      likes: 38,
      downloads: 96,
      tags: ['documents', 'search', 'metadata'],
      code: `// Create database for document embeddings
const db = await createDatabase({
  name: 'document_embeddings',
  dimensions: 512
});

// Add document embeddings
const documents = [
  { id: 'doc1', vector: [/* embedding values */], metadata: { title: 'AI Research Paper', category: 'research' } },
  // ...
];

for (const doc of documents) {
  await addVector(db.id, doc);
}

// Semantic search
const query = "machine learning algorithms";
const queryEmbedding = await getEmbedding(query);
const results = await similaritySearch(db.id, queryEmbedding, { 
  top_k: 10,
  filters: { category: 'research' }
});`,
      createdAt: '2023-10-12',
      stars: 4.5
    },
    {
      id: '3',
      title: 'Image Similarity Engine',
      description: 'Find similar images using visual embeddings',
      author: 'vision_dev',
      likes: 67,
      downloads: 187,
      tags: ['images', 'similarity', 'computer-vision'],
      code: `// Create image embedding database
const db = await createDatabase({
  name: 'image_embeddings',
  dimensions: 2048
});

// Add image embeddings
const images = [
  { id: 'img1', vector: [/* visual embedding */], metadata: { path: '/path/to/image1.jpg', tags: ['cat', 'pet'] } },
  // ...
];

for (const image of images) {
  await addVector(db.id, image);
}

// Find similar images
const targetImageEmbedding = await getEmbeddingFromImage('/path/to/target.jpg');
const results = await similaritySearch(db.id, targetImageEmbedding, { 
  top_k: 10,
  threshold: 0.8 
});`,
      createdAt: '2023-10-15',
      stars: 4.9
    }
  ]);

  const [mySharedScenarios, setMySharedScenarios] = useState([
    {
      id: 'my1',
      title: 'Custom Search Algorithm',
      description: 'My custom search algorithm with special filtering',
      likes: 12,
      downloads: 34,
      tags: ['search', 'custom', 'filtering'],
      createdAt: '2023-10-05',
      stars: 4.2
    }
  ]);

  const handleShare = () => {
    if (title.trim() && code.trim()) {
      // In a real implementation, this would upload the scenario to a backend
      const newScenario = {
        id: `new_${Date.now()}`,
        title,
        description,
        author: 'current_user', // Would be from authentication
        likes: 0,
        downloads: 0,
        tags: tags.split(',').map(tag => tag.trim()).filter(tag => tag),
        code,
        createdAt: new Date().toISOString().split('T')[0],
        stars: 0
      };
      
      setMySharedScenarios([...mySharedScenarios, newScenario]);
      
      // Reset form
      setTitle('');
      setDescription('');
      setCode('');
      setTags('');
      
      setActiveTab('my-shared');
    }
  };

  const handleLike = (id, section) => {
    // In a real implementation, this would update the backend
    if (section === 'community') {
      // Update communityScenarios with new like count
    } else if (section === 'my-shared') {
      // Update mySharedScenarios with new like count
    }
  };

  const handleDownload = (id) => {
    // In a real implementation, this would download the scenario
    console.log(`Downloading scenario ${id}`);
  };

  const copyCodeToClipboard = (code, id) => {
    navigator.clipboard.writeText(code);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const toggleCodeVisibility = (id) => {
    setShowCode(prev => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  const filteredCommunityScenarios = communityScenarios.filter(scenario => 
    scenario.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    scenario.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
    scenario.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  return (
    <div className="border border-gray-200 rounded-lg p-4 h-full overflow-hidden flex flex-col">
      <div className="pb-3 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-800 flex items-center gap-2">
          <span>📤</span>
          Community Sharing
        </h3>
        <p className="text-sm text-gray-600 mt-1">
          Share your scenarios or explore what others have created
        </p>
      </div>
      
      <div className="flex-1 flex flex-col gap-4 pt-4 overflow-hidden">
        <div className="flex border-b border-gray-200">
          <button
            className={`px-4 py-2 text-sm font-medium rounded-t-lg ${
              activeTab === 'share' 
                ? 'bg-white border-t border-l border-r border-gray-200 text-gray-800' 
                : 'text-gray-600 hover:text-gray-800'
            }`}
            onClick={() => setActiveTab('share')}
          >
            <span className="flex items-center gap-1">
              <span>📤</span> Share
            </span>
          </button>
          <button
            className={`px-4 py-2 text-sm font-medium rounded-t-lg ${
              activeTab === 'community' 
                ? 'bg-white border-t border-l border-r border-gray-200 text-gray-800' 
                : 'text-gray-600 hover:text-gray-800'
            }`}
            onClick={() => setActiveTab('community')}
          >
            <span className="flex items-center gap-1">
              <span>👥</span> Community
            </span>
          </button>
          <button
            className={`px-4 py-2 text-sm font-medium rounded-t-lg ${
              activeTab === 'my-shared' 
                ? 'bg-white border-t border-l border-r border-gray-200 text-gray-800' 
                : 'text-gray-600 hover:text-gray-800'
            }`}
            onClick={() => setActiveTab('my-shared')}
          >
            <span className="flex items-center gap-1">
              <span>💻</span> My Shared
            </span>
          </button>
        </div>
        
        <div className="flex-1 overflow-y-auto">
          {activeTab === 'share' && (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Title</label>
                <input 
                  type="text"
                  value={title} 
                  onChange={(e) => setTitle(e.target.value)} 
                  placeholder="Enter a descriptive title for your scenario" 
                  className="w-full p-2 border border-gray-300 rounded-md"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
                <textarea 
                  value={description} 
                  onChange={(e) => setDescription(e.target.value)} 
                  placeholder="Describe what your scenario does and how it works" 
                  rows={3}
                  className="w-full p-2 border border-gray-300 rounded-md"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Tags</label>
                <input 
                  type="text"
                  value={tags} 
                  onChange={(e) => setTags(e.target.value)} 
                  placeholder="Enter tags separated by commas (e.g. search, similarity, embedding)" 
                  className="w-full p-2 border border-gray-300 rounded-md"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Code Snippet</label>
                <textarea 
                  value={code} 
                  onChange={(e) => setCode(e.target.value)} 
                  placeholder="Paste your code here..." 
                  rows={8}
                  className="w-full p-2 border border-gray-300 rounded-md font-mono text-sm"
                />
              </div>
              
              <div className="flex justify-end">
                <button 
                  onClick={handleShare}
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 flex items-center gap-2"
                >
                  <span>📤</span> Share Scenario
                </button>
              </div>
            </div>
          )}
          
          {activeTab === 'community' && (
            <div className="flex flex-col h-full">
              <div className="mb-4 relative">
                <input 
                  type="text"
                  value={searchQuery} 
                  onChange={(e) => setSearchQuery(e.target.value)} 
                  placeholder="Search scenarios by title, description, or tags..." 
                  className="w-full p-2 pl-8 border border-gray-300 rounded-md"
                />
                <span className="absolute left-2 top-1/2 transform -translate-y-1/2 text-gray-400">🔍</span>
              </div>
              
              <div className="flex-1 overflow-y-auto space-y-4">
                {filteredCommunityScenarios.map(scenario => (
                  <div key={scenario.id} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                    <div className="flex justify-between items-start">
                      <div>
                        <h3 className="font-semibold text-gray-800">{scenario.title}</h3>
                        <p className="text-sm text-gray-600 mt-1">{scenario.description}</p>
                        <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
                          <span>by {scenario.author}</span>
                          <div className="flex items-center">
                            {[...Array(5)].map((_, i) => (
                              <span key={i} className={i < Math.floor(scenario.stars) ? 'text-yellow-500' : 'text-gray-300'}>
                                {i < Math.floor(scenario.stars) ? '★' : '☆'}
                              </span>
                            ))}
                            <span className="ml-1">{scenario.stars}</span>
                          </div>
                        </div>
                        
                        <div className="flex flex-wrap gap-1 mt-2">
                          {scenario.tags.map((tag, idx) => (
                            <span key={idx} className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                              {tag}
                            </span>
                          ))}
                        </div>
                      </div>
                      
                      <div className="flex flex-col items-end">
                        <span className="text-xs text-gray-500">{scenario.createdAt}</span>
                        <div className="flex items-center gap-2 mt-2">
                          <button 
                            className="text-sm flex items-center gap-1 px-2 py-1 text-gray-600 hover:bg-gray-100 rounded"
                            onClick={() => handleLike(scenario.id, 'community')}
                          >
                            <span>❤️</span> {scenario.likes}
                          </button>
                          <button 
                            className="text-sm flex items-center gap-1 px-2 py-1 text-gray-600 hover:bg-gray-100 rounded"
                            onClick={() => handleDownload(scenario.id)}
                          >
                            <span>⬇️</span> {scenario.downloads}
                          </button>
                          <button 
                            className="text-sm flex items-center gap-1 px-2 py-1 text-gray-600 hover:bg-gray-100 rounded"
                            onClick={() => toggleCodeVisibility(scenario.id)}
                          >
                            {showCode[scenario.id] ? '👁️' : '👁️‍🗨️'}
                          </button>
                        </div>
                      </div>
                    </div>
                    
                    {showCode[scenario.id] && (
                      <div className="mt-4 relative">
                        <pre className="bg-gray-100 rounded-md p-4 text-xs overflow-x-auto">
                          <code>{scenario.code}</code>
                        </pre>
                        <button 
                          className="absolute top-2 right-2 px-2 py-1 bg-gray-200 hover:bg-gray-300 text-gray-700 rounded text-sm"
                          onClick={() => copyCodeToClipboard(scenario.code, scenario.id)}
                        >
                          {copiedId === scenario.id ? '✓ Copied' : '📋 Copy'}
                        </button>
                      </div>
                    )}
                  </div>
                ))}
                
                {filteredCommunityScenarios.length === 0 && (
                  <div className="text-center py-8 text-gray-500">
                    No scenarios found matching your search.
                  </div>
                )}
              </div>
            </div>
          )}
          
          {activeTab === 'my-shared' && (
            <div className="flex-1 overflow-y-auto space-y-4">
              {mySharedScenarios.map(scenario => (
                <div key={scenario.id} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                  <div className="flex justify-between items-start">
                    <div>
                      <h3 className="font-semibold text-gray-800">{scenario.title}</h3>
                      <p className="text-sm text-gray-600 mt-1">{scenario.description}</p>
                      
                      <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
                        <div className="flex items-center">
                          {[...Array(5)].map((_, i) => (
                            <span key={i} className={i < Math.floor(scenario.stars) ? 'text-yellow-500' : 'text-gray-300'}>
                              {i < Math.floor(scenario.stars) ? '★' : '☆'}
                            </span>
                          ))}
                          <span className="ml-1">{scenario.stars}</span>
                        </div>
                      </div>
                      
                      <div className="flex flex-wrap gap-1 mt-2">
                        {scenario.tags.map((tag, idx) => (
                          <span key={idx} className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>
                    
                    <div className="flex flex-col items-end">
                      <span className="text-xs text-gray-500">{scenario.createdAt}</span>
                      <div className="flex items-center gap-2 mt-2">
                        <button 
                          className="text-sm flex items-center gap-1 px-2 py-1 text-gray-600 hover:bg-gray-100 rounded"
                          onClick={() => handleLike(scenario.id, 'my-shared')}
                        >
                          <span>❤️</span> {scenario.likes}
                        </button>
                        <button 
                          className="text-sm flex items-center gap-1 px-2 py-1 text-gray-600 hover:bg-gray-100 rounded"
                          onClick={() => handleDownload(scenario.id)}
                        >
                          <span>⬇️</span> {scenario.downloads}
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
              
              {mySharedScenarios.length === 0 && (
                <div className="text-center py-8 text-gray-500">
                  You haven't shared any scenarios yet. Create one in the Share tab!
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CommunitySharing;