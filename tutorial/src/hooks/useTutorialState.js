import { useState, useEffect, useCallback, useMemo } from 'react';
import { useAdvancedTutorial } from '../contexts/AdvancedTutorialContext';

/**
 * Custom hook for managing tutorial state with additional utilities
 */
export const useTutorialState = () => {
  const { state, actions } = useAdvancedTutorial();
  
  // Current module and step
  const currentModule = useMemo(() => state.modules[state.currentModule] || null, [state]);
  const currentStep = useMemo(() => state.currentModule, [state]);
  
  // Progress tracking
  const overallProgress = useMemo(() => actions.getOverallProgress(), [actions]);
  
  const moduleProgress = useCallback((moduleId) => {
    return actions.getModuleProgress(moduleId);
  }, [actions]);
  
  // Navigation functions
  const goToNextStep = useCallback(() => {
    if (state.currentStep < (currentModule?.steps - 1 || 0)) {
      actions.updateModuleProgress(state.currentModule, state.currentModule.completedSteps + 1);
    }
  }, [state, currentModule, actions]);
  
  const goToPreviousStep = useCallback(() => {
    if (state.currentStep > 0) {
      actions.updateModuleProgress(state.currentModule, state.currentModule.completedSteps - 1);
    }
  }, [state, actions]);
  
  const goToNextModule = useCallback(() => {
    if (state.currentModule < state.modules.length - 1) {
      // Update current module progress
      actions.updateModuleProgress(state.currentModule, state.modules[state.currentModule].steps);
      
      // Move to next module
      const newState = { ...state };
      newState.currentModule += 1;
      newState.currentStep = 0;
      actions.updateModuleProgress(newState.currentModule, 0);
    }
  }, [state, actions]);
  
  const goToPreviousModule = useCallback(() => {
    if (state.currentModule > 0) {
      // Move to previous module
      const newState = { ...state };
      newState.currentModule -= 1;
      newState.currentStep = 0;
      actions.updateModuleProgress(newState.currentModule, 0);
    }
  }, [state, actions]);
  
  // Module unlocking
  const unlockModule = useCallback((moduleId) => {
    if (moduleId >= 0 && moduleId < state.modules.length) {
      const newState = { ...state };
      newState.modules[moduleId].unlocked = true;
      actions.updateModuleProgress(moduleId, newState.modules[moduleId].completedSteps);
    }
  }, [state, actions]);
  
  // Preferences management
  const updatePreference = useCallback((key, value) => {
    const newPreferences = { ...state.preferences, [key]: value };
    actions.updatePreferences(newPreferences);
  }, [state, actions]);
  
  // Achievement and badge management
  const addAchievementIfNotExists = useCallback((achievement) => {
    const exists = state.achievements.some(a => a.id === achievement.id);
    if (!exists) {
      actions.addAchievement(achievement);
    }
  }, [state, actions]);
  
  const addBadgeIfNotExists = useCallback((badge) => {
    const exists = state.badges.some(b => b.id === badge.id);
    if (!exists) {
      actions.addBadge(badge);
    }
  }, [state, actions]);
  
  // Code snippet management
  const saveSnippet = useCallback((snippet) => {
    actions.saveCodeSnippet(snippet);
  }, [actions]);
  
  const deleteSnippet = useCallback((snippetId) => {
    actions.deleteCodeSnippet(snippetId);
  }, [actions]);
  
  // Assessment results
  const saveAssessment = useCallback((moduleId, step, result) => {
    actions.saveAssessmentResult(moduleId, step, result);
  }, [actions]);
  
  // Time tracking
  const [timeSpent, setTimeSpent] = useState(0);
  
  useEffect(() => {
    const interval = setInterval(() => {
      setTimeSpent(prev => prev + 1);
      actions.updateTimeSpent(state.currentModule, 1);
    }, 1000);
    
    return () => clearInterval(interval);
  }, [state.currentModule, actions]);
  
  // Statistics
  const statistics = useMemo(() => actions.getStatistics(), [actions]);
  
  // Bookmarks
  const addBookmark = useCallback((bookmark) => {
    actions.addBookmark(bookmark);
  }, [actions]);
  
  const removeBookmark = useCallback((bookmarkUrl) => {
    actions.removeBookmark(bookmarkUrl);
  }, [actions]);
  
  // Search history
  const addToSearchHistory = useCallback((searchTerm) => {
    actions.addSearchToHistory(searchTerm);
  }, [actions]);
  
  // Feedback
  const submitFeedback = useCallback((feedback) => {
    actions.addFeedback(feedback);
  }, [actions]);
  
  // Export/Import
  const exportTutorialProgress = useCallback(() => {
    return actions.exportProgress();
  }, [actions]);
  
  const importTutorialProgress = useCallback((progressData) => {
    return actions.importProgress(progressData);
  }, [actions]);
  
  // Reset progress
  const resetAllProgress = useCallback(() => {
    actions.resetProgress();
  }, [actions]);
  
  return {
    // State
    state,
    currentModule,
    currentStep,
    timeSpent,
    statistics,
    
    // Progress
    overallProgress,
    moduleProgress,
    
    // Navigation
    goToNextStep,
    goToPreviousStep,
    goToNextModule,
    goToPreviousModule,
    
    // Module management
    unlockModule,
    
    // Preferences
    updatePreference,
    
    // Achievements and badges
    addAchievementIfNotExists,
    addBadgeIfNotExists,
    
    // Code snippets
    saveSnippet,
    deleteSnippet,
    
    // Assessments
    saveAssessment,
    
    // Bookmarks
    addBookmark,
    removeBookmark,
    
    // Search
    addToSearchHistory,
    
    // Feedback
    submitFeedback,
    
    // Export/Import
    exportTutorialProgress,
    importTutorialProgress,
    
    // Reset
    resetAllProgress
  };
};