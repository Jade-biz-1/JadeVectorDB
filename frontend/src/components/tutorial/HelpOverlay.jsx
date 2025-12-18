import React, { useState, useEffect } from 'react';
import { useContextualHelp } from '../../hooks/useContextualHelp';

/**
 * HelpOverlay - Full-screen help system with search and navigation
 */
const HelpOverlay = ({ isOpen, onClose, initialContext = null }) => {
  const {
    searchQuery,
    setSearchQuery,
    searchHelp,
    getHelpTopic,
    getRelatedTopics,
    getTopicsByCategory,
    getContextualHelp,
    categories,
    quickTips
  } = useContextualHelp();

  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [searchResults, setSearchResults] = useState([]);

  // Load initial context or show all topics
  useEffect(() => {
    if (isOpen) {
      if (initialContext) {
        const contextHelp = getContextualHelp(initialContext);
        if (contextHelp) {
          // Show contextual help
          setSelectedTopic({ id: initialContext, ...contextHelp, isContextual: true });
        }
      } else {
        // Show all topics
        setSearchResults(searchHelp(''));
      }
    }
  }, [isOpen, initialContext, getContextualHelp, searchHelp]);

  // Handle search
  useEffect(() => {
    const results = searchHelp(searchQuery);
    setSearchResults(results);
  }, [searchQuery, searchHelp]);

  // Filter by category
  const filteredResults = selectedCategory === 'all'
    ? searchResults
    : searchResults.filter(topic => topic.category === selectedCategory);

  // Handle topic selection
  const handleTopicClick = (topic) => {
    setSelectedTopic(topic);
  };

  // Handle close
  const handleClose = () => {
    setSearchQuery('');
    setSelectedTopic(null);
    setSelectedCategory('all');
    if (onClose) onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4">
      <div className="bg-white rounded-lg shadow-2xl w-full max-w-6xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-6 rounded-t-lg flex-shrink-0">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <span className="text-4xl">📚</span>
              <div>
                <h2 className="text-3xl font-bold">Help Center</h2>
                <p className="text-blue-100 text-sm">
                  Press F1 or ? anytime to open help • ESC to close
                </p>
              </div>
            </div>
            <button
              onClick={handleClose}
              className="text-white hover:text-gray-200 text-3xl leading-none"
              aria-label="Close help"
            >
              ×
            </button>
          </div>

          {/* Search Bar */}
          <div className="relative">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search help topics..."
              className="w-full px-4 py-3 rounded-lg text-gray-800 placeholder-gray-400 pr-10"
              autoFocus
            />
            <span className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 text-xl">
              🔍
            </span>
          </div>
        </div>

        {/* Content */}
        <div className="flex flex-1 overflow-hidden">
          {/* Sidebar */}
          <div className="w-64 bg-gray-50 border-r overflow-y-auto flex-shrink-0">
            {/* Categories */}
            <div className="p-4">
              <h3 className="font-bold text-gray-700 mb-2">Categories</h3>
              <div className="space-y-1">
                <button
                  onClick={() => setSelectedCategory('all')}
                  className={`w-full text-left px-3 py-2 rounded transition-colors ${
                    selectedCategory === 'all'
                      ? 'bg-blue-600 text-white'
                      : 'hover:bg-gray-200 text-gray-700'
                  }`}
                >
                  All Topics
                </button>
                {categories.map(category => (
                  <button
                    key={category}
                    onClick={() => setSelectedCategory(category)}
                    className={`w-full text-left px-3 py-2 rounded transition-colors ${
                      selectedCategory === category
                        ? 'bg-blue-600 text-white'
                        : 'hover:bg-gray-200 text-gray-700'
                    }`}
                  >
                    {category}
                  </button>
                ))}
              </div>
            </div>

            {/* Quick Tips */}
            <div className="p-4 border-t">
              <h3 className="font-bold text-gray-700 mb-2">💡 Quick Tip</h3>
              <div className="text-sm text-gray-600 bg-yellow-50 border border-yellow-200 rounded p-3">
                {quickTips[Math.floor(Date.now() / 10000) % quickTips.length]}
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="flex-1 overflow-y-auto">
            {selectedTopic ? (
              // Topic Detail View
              <div className="p-6">
                <button
                  onClick={() => setSelectedTopic(null)}
                  className="text-blue-600 hover:text-blue-700 mb-4 flex items-center gap-2"
                >
                  ← Back to topics
                </button>

                <div className="bg-white rounded-lg border p-6">
                  <div className="flex items-start gap-3 mb-4">
                    <span className="text-4xl">{selectedTopic.icon || '📄'}</span>
                    <div>
                      <h3 className="text-2xl font-bold text-gray-800">
                        {selectedTopic.title}
                      </h3>
                      {!selectedTopic.isContextual && (
                        <span className="inline-block mt-2 px-3 py-1 bg-blue-100 text-blue-800 text-sm font-semibold rounded">
                          {selectedTopic.category}
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="prose max-w-none">
                    <p className="text-gray-700 leading-relaxed text-lg">
                      {selectedTopic.content}
                    </p>
                  </div>

                  {!selectedTopic.isContextual && selectedTopic.keywords && (
                    <div className="mt-6 pt-6 border-t">
                      <h4 className="font-semibold text-gray-700 mb-2">Keywords</h4>
                      <div className="flex flex-wrap gap-2">
                        {selectedTopic.keywords.map((keyword, index) => (
                          <span
                            key={index}
                            className="px-3 py-1 bg-gray-100 text-gray-700 text-sm rounded"
                          >
                            {keyword}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {!selectedTopic.isContextual && selectedTopic.relatedTopics && selectedTopic.relatedTopics.length > 0 && (
                    <div className="mt-6 pt-6 border-t">
                      <h4 className="font-semibold text-gray-700 mb-3">Related Topics</h4>
                      <div className="space-y-2">
                        {getRelatedTopics(selectedTopic.id).map(related => (
                          <button
                            key={related.id}
                            onClick={() => handleTopicClick(related)}
                            className="w-full text-left px-4 py-3 bg-gray-50 hover:bg-gray-100 rounded-lg transition-colors"
                          >
                            <div className="font-medium text-blue-600">{related.title}</div>
                            <div className="text-sm text-gray-600">{related.category}</div>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              // Topic List View
              <div className="p-6">
                <div className="mb-4 flex items-center justify-between">
                  <h3 className="text-xl font-bold text-gray-800">
                    {searchQuery
                      ? `Search results for "${searchQuery}"`
                      : selectedCategory === 'all'
                      ? 'All Help Topics'
                      : `${selectedCategory} Topics`}
                  </h3>
                  <span className="text-gray-600">
                    {filteredResults.length} topic{filteredResults.length !== 1 ? 's' : ''}
                  </span>
                </div>

                {filteredResults.length === 0 ? (
                  <div className="text-center py-12">
                    <span className="text-6xl mb-4 block">🔍</span>
                    <p className="text-xl text-gray-600">No topics found</p>
                    <p className="text-gray-500 mt-2">Try a different search term</p>
                  </div>
                ) : (
                  <div className="grid gap-4">
                    {filteredResults.map(topic => (
                      <button
                        key={topic.id}
                        onClick={() => handleTopicClick(topic)}
                        className="text-left p-4 bg-white border border-gray-200 rounded-lg hover:border-blue-500 hover:shadow-md transition-all"
                      >
                        <div className="flex items-start gap-3">
                          <div className="flex-1">
                            <h4 className="font-bold text-gray-800 text-lg mb-1">
                              {topic.title}
                            </h4>
                            <p className="text-gray-600 text-sm mb-2 line-clamp-2">
                              {topic.content}
                            </p>
                            <span className="inline-block px-2 py-1 bg-gray-100 text-gray-700 text-xs font-semibold rounded">
                              {topic.category}
                            </span>
                          </div>
                          <span className="text-gray-400">→</span>
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="bg-gray-50 px-6 py-4 border-t flex-shrink-0 rounded-b-lg">
          <div className="flex items-center justify-between text-sm text-gray-600">
            <div className="flex items-center gap-4">
              <span>💡 Tip: Use the search bar to find specific topics</span>
            </div>
            <button
              onClick={handleClose}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-semibold"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HelpOverlay;
