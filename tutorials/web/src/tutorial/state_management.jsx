import React, { createContext, useContext, useReducer, useEffect, useCallback } from 'react';
import advancedTutorialState from '../../lib/advancedTutorialState';

// Define action types
const ACTIONS = {
  SET_STATE: 'SET_STATE',
  UPDATE_MODULE_PROGRESS: 'UPDATE_MODULE_PROGRESS',
  UPDATE_PREFERENCES: 'UPDATE_PREFERENCES',
  ADD_ACHIEVEMENT: 'ADD_ACHIEVEMENT',
  ADD_BADGE: 'ADD_BADGE',
  RESET_PROGRESS: 'RESET_PROGRESS',
  SAVE_CODE_SNIPPET: 'SAVE_CODE_SNIPPET',
  DELETE_CODE_SNIPPET: 'DELETE_CODE_SNIPPET',
  SAVE_ASSESSMENT_RESULT: 'SAVE_ASSESSMENT_RESULT',
  ADD_CUSTOM_SCENARIO: 'ADD_CUSTOM_SCENARIO',
  ADD_BOOKMARK: 'ADD_BOOKMARK',
  REMOVE_BOOKMARK: 'REMOVE_BOOKMARK',
  ADD_SEARCH_HISTORY: 'ADD_SEARCH_HISTORY',
  ADD_FEEDBACK: 'ADD_FEEDBACK',
  UPDATE_TIME_SPENT: 'UPDATE_TIME_SPENT'
};

// Initial state
const initialState = advancedTutorialState.getInitialState();

// Reducer function
const tutorialReducer = (state, action) => {
  switch (action.type) {
    case ACTIONS.SET_STATE:
      return { ...state, ...action.payload };
      
    case ACTIONS.UPDATE_MODULE_PROGRESS:
      return advancedTutorialState.updateModuleProgress(action.moduleId, action.completedSteps);
      
    case ACTIONS.UPDATE_PREFERENCES:
      return advancedTutorialState.updatePreferences(action.preferences);
      
    case ACTIONS.ADD_ACHIEVEMENT:
      return advancedTutorialState.addAchievement(action.achievement);
      
    case ACTIONS.ADD_BADGE:
      return advancedTutorialState.addBadge(action.badge);
      
    case ACTIONS.RESET_PROGRESS:
      return advancedTutorialState.resetProgress();
      
    case ACTIONS.SAVE_CODE_SNIPPET:
      return advancedTutorialState.saveCodeSnippet(action.snippet);
      
    case ACTIONS.DELETE_CODE_SNIPPET:
      return advancedTutorialState.deleteCodeSnippet(action.snippetId);
      
    case ACTIONS.SAVE_ASSESSMENT_RESULT:
      return advancedTutorialState.saveAssessmentResult(action.moduleId, action.step, action.result);
      
    case ACTIONS.ADD_CUSTOM_SCENARIO:
      return advancedTutorialState.addCustomScenario(action.scenario);
      
    case ACTIONS.ADD_BOOKMARK:
      return advancedTutorialState.addBookmark(action.bookmark);
      
    case ACTIONS.REMOVE_BOOKMARK:
      return advancedTutorialState.removeBookmark(action.bookmarkUrl);
      
    case ACTIONS.ADD_SEARCH_HISTORY:
      return advancedTutorialState.addSearchToHistory(action.searchTerm);
      
    case ACTIONS.ADD_FEEDBACK:
      return advancedTutorialState.addFeedback(action.feedback);
      
    case ACTIONS.UPDATE_TIME_SPENT:
      return advancedTutorialState.updateTimeSpent(action.moduleId, action.seconds);
      
    default:
      return state;
  }
};

// Create context
const TutorialStateContext = createContext();

// Provider component
export const TutorialStateProvider = ({ children }) => {
  const [state, dispatch] = useReducer(tutorialReducer, initialState);
  
  // Load state from localStorage on mount
  useEffect(() => {
    const savedState = advancedTutorialState.loadFromStorage();
    if (savedState) {
      dispatch({ type: ACTIONS.SET_STATE, payload: savedState });
    }
  }, []);
  
  // Save state to localStorage whenever it changes
  useEffect(() => {
    advancedTutorialState.saveToStorage(state);
  }, [state]);
  
  // Action creators
  const updateModuleProgress = useCallback((moduleId, completedSteps) => {
    dispatch({ type: ACTIONS.UPDATE_MODULE_PROGRESS, moduleId, completedSteps });
  }, []);
  
  const updatePreferences = useCallback((preferences) => {
    dispatch({ type: ACTIONS.UPDATE_PREFERENCES, preferences });
  }, []);
  
  const addAchievement = useCallback((achievement) => {
    dispatch({ type: ACTIONS.ADD_ACHIEVEMENT, achievement });
  }, []);
  
  const addBadge = useCallback((badge) => {
    dispatch({ type: ACTIONS.ADD_BADGE, badge });
  }, []);
  
  const resetProgress = useCallback(() => {
    dispatch({ type: ACTIONS.RESET_PROGRESS });
  }, []);
  
  const saveCodeSnippet = useCallback((snippet) => {
    dispatch({ type: ACTIONS.SAVE_CODE_SNIPPET, snippet });
  }, []);
  
  const deleteCodeSnippet = useCallback((snippetId) => {
    dispatch({ type: ACTIONS.DELETE_CODE_SNIPPET, snippetId });
  }, []);
  
  const saveAssessmentResult = useCallback((moduleId, step, result) => {
    dispatch({ type: ACTIONS.SAVE_ASSESSMENT_RESULT, moduleId, step, result });
  }, []);
  
  const addCustomScenario = useCallback((scenario) => {
    dispatch({ type: ACTIONS.ADD_CUSTOM_SCENARIO, scenario });
  }, []);
  
  const addBookmark = useCallback((bookmark) => {
    dispatch({ type: ACTIONS.ADD_BOOKMARK, bookmark });
  }, []);
  
  const removeBookmark = useCallback((bookmarkUrl) => {
    dispatch({ type: ACTIONS.REMOVE_BOOKMARK, bookmarkUrl });
  }, []);
  
  const addSearchToHistory = useCallback((searchTerm) => {
    dispatch({ type: ACTIONS.ADD_SEARCH_HISTORY, searchTerm });
  }, []);
  
  const addFeedback = useCallback((feedback) => {
    dispatch({ type: ACTIONS.ADD_FEEDBACK, feedback });
  }, []);
  
  const updateTimeSpent = useCallback((moduleId, seconds) => {
    dispatch({ type: ACTIONS.UPDATE_TIME_SPENT, moduleId, seconds });
  }, []);
  
  // Calculate progress percentages
  const getOverallProgress = useCallback(() => {
    return advancedTutorialState.getProgressPercentage();
  }, []);
  
  const getModuleProgress = useCallback((moduleId) => {
    return advancedTutorialState.getModuleProgressPercentage(moduleId);
  }, []);
  
  // Get statistics
  const getStatistics = useCallback(() => {
    return advancedTutorialState.getStatistics();
  }, []);
  
  // Export/Import progress
  const exportProgress = useCallback(() => {
    return advancedTutorialState.exportProgress();
  }, []);
  
  const importProgress = useCallback((progressData) => {
    return advancedTutorialState.importProgress(progressData);
  }, []);
  
  // Context value
  const value = {
    state,
    actions: {
      updateModuleProgress,
      updatePreferences,
      addAchievement,
      addBadge,
      resetProgress,
      saveCodeSnippet,
      deleteCodeSnippet,
      saveAssessmentResult,
      addCustomScenario,
      addBookmark,
      removeBookmark,
      addSearchToHistory,
      addFeedback,
      updateTimeSpent,
      getOverallProgress,
      getModuleProgress,
      getStatistics,
      exportProgress,
      importProgress
    }
  };
  
  return (
    <TutorialStateContext.Provider value={value}>
      {children}
    </TutorialStateContext.Provider>
  );
};

// Hook to use the tutorial state context
export const useTutorialStateContext = () => {
  const context = useContext(TutorialStateContext);
  
  if (!context) {
    throw new Error('useTutorialStateContext must be used within a TutorialStateProvider');
  }
  
  return context;
};

export default TutorialStateContext;