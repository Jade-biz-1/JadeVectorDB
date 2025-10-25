import AdvancedTutorialState from './advancedTutorialState';
import { AdvancedTutorialProvider } from '../contexts/AdvancedTutorialContext';

/**
 * Tutorial State Management System
 * 
 * This module provides a comprehensive state management solution for the 
 * JadeVectorDB interactive tutorial, including progress tracking, preferences,
 * achievements, and more.
 */

class TutorialStateManager {
  constructor() {
    this.advancedState = AdvancedTutorialState;
    this.provider = AdvancedTutorialProvider;
  }
  
  /**
   * Get the initial state for the tutorial
   * @returns {Object} Initial state object
   */
  getInitialState() {
    return this.advancedState.getInitialState();
  }
  
  /**
   * Save state to persistent storage
   * @param {Object} state - State to save
   * @returns {boolean} Success status
   */
  saveToStorage(state) {
    return this.advancedState.saveToStorage(state);
  }
  
  /**
   * Load state from persistent storage
   * @returns {Object|null} Loaded state or null if not found
   */
  loadFromStorage() {
    return this.advancedState.loadFromStorage();
  }
  
  /**
   * Clear all saved state
   * @returns {boolean} Success status
   */
  clearStorage() {
    return this.advancedState.clearStorage();
  }
  
  /**
   * Update progress for a specific module
   * @param {number} moduleId - Module ID
   * @param {number} completedSteps - Number of completed steps
   * @returns {Object} Updated state
   */
  updateModuleProgress(moduleId, completedSteps) {
    return this.advancedState.updateModuleProgress(moduleId, completedSteps);
  }
  
  /**
   * Update user preferences
   * @param {Object} preferences - Preferences to update
   * @returns {Object} Updated state
   */
  updatePreferences(preferences) {
    return this.advancedState.updatePreferences(preferences);
  }
  
  /**
   * Add a new achievement
   * @param {Object} achievement - Achievement to add
   * @returns {Object} Updated state
   */
  addAchievement(achievement) {
    return this.advancedState.addAchievement(achievement);
  }
  
  /**
   * Add a new badge
   * @param {Object} badge - Badge to add
   * @returns {Object} Updated state
   */
  addBadge(badge) {
    return this.advancedState.addBadge(badge);
  }
  
  /**
   * Reset all tutorial progress
   * @returns {Object} Initial state
   */
  resetProgress() {
    return this.advancedState.resetProgress();
  }
  
  /**
   * Get overall progress percentage
   * @returns {number} Progress percentage (0-100)
   */
  getProgressPercentage() {
    return this.advancedState.getProgressPercentage();
  }
  
  /**
   * Get progress percentage for a specific module
   * @param {number} moduleId - Module ID
   * @returns {number} Progress percentage (0-100)
   */
  getModuleProgressPercentage(moduleId) {
    return this.advancedState.getModuleProgressPercentage(moduleId);
  }
  
  /**
   * Save a code snippet
   * @param {Object} snippet - Code snippet to save
   * @returns {Object} Updated state
   */
  saveCodeSnippet(snippet) {
    return this.advancedState.saveCodeSnippet(snippet);
  }
  
  /**
   * Delete a code snippet
   * @param {string} snippetId - ID of snippet to delete
   * @returns {Object} Updated state
   */
  deleteCodeSnippet(snippetId) {
    return this.advancedState.deleteCodeSnippet(snippetId);
  }
  
  /**
   * Save assessment result
   * @param {number} moduleId - Module ID
   * @param {number} step - Step number
   * @param {Object} result - Assessment result
   * @returns {Object} Updated state
   */
  saveAssessmentResult(moduleId, step, result) {
    return this.advancedState.saveAssessmentResult(moduleId, step, result);
  }
  
  /**
   * Add a custom scenario
   * @param {Object} scenario - Custom scenario to add
   * @returns {Object} Updated state
   */
  addCustomScenario(scenario) {
    return this.advancedState.addCustomScenario(scenario);
  }
  
  /**
   * Add a bookmark
   * @param {Object} bookmark - Bookmark to add
   * @returns {Object} Updated state
   */
  addBookmark(bookmark) {
    return this.advancedState.addBookmark(bookmark);
  }
  
  /**
   * Remove a bookmark
   * @param {string} bookmarkUrl - URL of bookmark to remove
   * @returns {Object} Updated state
   */
  removeBookmark(bookmarkUrl) {
    return this.advancedState.removeBookmark(bookmarkUrl);
  }
  
  /**
   * Add search term to history
   * @param {string} searchTerm - Search term to add
   * @returns {Object} Updated state
   */
  addSearchToHistory(searchTerm) {
    return this.advancedState.addSearchToHistory(searchTerm);
  }
  
  /**
   * Add user feedback
   * @param {Object} feedback - Feedback to add
   * @returns {Object} Updated state
   */
  addFeedback(feedback) {
    return this.advancedState.addFeedback(feedback);
  }
  
  /**
   * Update time spent on a module
   * @param {number} moduleId - Module ID
   * @param {number} seconds - Seconds to add
   * @returns {Object} Updated state
   */
  updateTimeSpent(moduleId, seconds) {
    return this.advancedState.updateTimeSpent(moduleId, seconds);
  }
  
  /**
   * Get session duration
   * @returns {number} Session duration in milliseconds
   */
  getSessionDuration() {
    return this.advancedState.getSessionDuration();
  }
  
  /**
   * Export progress data
   * @returns {string} JSON string of progress data
   */
  exportProgress() {
    return this.advancedState.exportProgress();
  }
  
  /**
   * Import progress data
   * @param {string} progressData - JSON string of progress data
   * @returns {boolean} Success status
   */
  importProgress(progressData) {
    return this.advancedState.importProgress(progressData);
  }
  
  /**
   * Get statistics for dashboard
   * @returns {Object} Statistics object
   */
  getStatistics() {
    return this.advancedState.getStatistics();
  }
  
  /**
   * Get the React context provider
   * @returns {React.Component} Context provider component
   */
  getProvider() {
    return this.provider;
  }
}

// Export singleton instance
const tutorialState = new TutorialStateManager();
export default tutorialState;

// Export individual components for direct access
export { AdvancedTutorialProvider, useAdvancedTutorial } from '../contexts/AdvancedTutorialContext';
export { useTutorialState } from '../hooks/useTutorialState';