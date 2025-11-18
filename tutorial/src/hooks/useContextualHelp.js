import { useState, useEffect, useCallback } from 'react';
import helpData from '../data/helpContent.json';

/**
 * useContextualHelp - React hook for contextual help system
 *
 * Provides access to help content, search, and keyboard shortcuts
 */
export function useContextualHelp() {
  const [isHelpOpen, setIsHelpOpen] = useState(false);
  const [currentContext, setCurrentContext] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');

  // Keyboard shortcut handler
  useEffect(() => {
    const handleKeyPress = (event) => {
      // F1 or ? opens help
      if (event.key === 'F1' || (event.key === '?' && !event.target.matches('input, textarea'))) {
        event.preventDefault();
        setIsHelpOpen(true);
      }

      // ESC closes help
      if (event.key === 'Escape' && isHelpOpen) {
        event.preventDefault();
        setIsHelpOpen(false);
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [isHelpOpen]);

  // Open help with optional context
  const openHelp = useCallback((context = null) => {
    setCurrentContext(context);
    setIsHelpOpen(true);
  }, []);

  // Close help
  const closeHelp = useCallback(() => {
    setIsHelpOpen(false);
    setCurrentContext(null);
    setSearchQuery('');
  }, []);

  // Get contextual help for current context
  const getContextualHelp = useCallback((contextId) => {
    return helpData.contextual_help[contextId] || null;
  }, []);

  // Search help topics
  const searchHelp = useCallback((query) => {
    if (!query || query.trim() === '') {
      return helpData.help_topics;
    }

    const lowerQuery = query.toLowerCase();
    return helpData.help_topics.filter(topic => {
      return (
        topic.title.toLowerCase().includes(lowerQuery) ||
        topic.content.toLowerCase().includes(lowerQuery) ||
        topic.keywords.some(keyword => keyword.toLowerCase().includes(lowerQuery)) ||
        topic.category.toLowerCase().includes(lowerQuery)
      );
    });
  }, []);

  // Get help topic by ID
  const getHelpTopic = useCallback((topicId) => {
    return helpData.help_topics.find(topic => topic.id === topicId);
  }, []);

  // Get related topics
  const getRelatedTopics = useCallback((topicId) => {
    const topic = getHelpTopic(topicId);
    if (!topic || !topic.relatedTopics) return [];

    return topic.relatedTopics
      .map(id => getHelpTopic(id))
      .filter(t => t !== undefined);
  }, [getHelpTopic]);

  // Get topics by category
  const getTopicsByCategory = useCallback((category) => {
    if (category === 'all') return helpData.help_topics;
    return helpData.help_topics.filter(topic => topic.category === category);
  }, []);

  // Get random quick tip
  const getRandomTip = useCallback(() => {
    const tips = helpData.quick_tips;
    return tips[Math.floor(Math.random() * tips.length)];
  }, []);

  return {
    // State
    isHelpOpen,
    currentContext,
    searchQuery,

    // Actions
    openHelp,
    closeHelp,
    setSearchQuery,

    // Data access
    getContextualHelp,
    searchHelp,
    getHelpTopic,
    getRelatedTopics,
    getTopicsByCategory,
    getRandomTip,

    // Constants
    categories: helpData.categories,
    allTopics: helpData.help_topics,
    quickTips: helpData.quick_tips
  };
}

export default useContextualHelp;
