class AdvancedTutorialState {
  constructor() {
    this.defaultState = {
      // User progress tracking
      currentModule: 0,
      currentStep: 0,
      moduleProgress: {},
      
      // Module completion status
      modules: [
        {
          id: 0,
          title: "Getting Started",
          description: "Learn basic concepts and create your first vector database",
          steps: 3,
          completedSteps: 0,
          unlocked: true,
          completionDate: null,
          timeSpent: 0 // in seconds
        },
        {
          id: 1,
          title: "Vector Manipulation",
          description: "Learn CRUD operations for vectors",
          steps: 4,
          completedSteps: 0,
          unlocked: false,
          completionDate: null,
          timeSpent: 0
        },
        {
          id: 2,
          title: "Advanced Search",
          description: "Master similarity search with various metrics",
          steps: 5,
          completedSteps: 0,
          unlocked: false,
          completionDate: null,
          timeSpent: 0
        },
        {
          id: 3,
          title: "Metadata Filtering",
          description: "Combine semantic and structural search",
          steps: 3,
          completedSteps: 0,
          unlocked: false,
          completionDate: null,
          timeSpent: 0
        },
        {
          id: 4,
          title: "Index Management",
          description: "Understand and configure indexing strategies",
          steps: 4,
          completedSteps: 0,
          unlocked: false,
          completionDate: null,
          timeSpent: 0
        },
        {
          id: 5,
          title: "Advanced Features",
          description: "Explore advanced capabilities",
          steps: 3,
          completedSteps: 0,
          unlocked: false,
          completionDate: null,
          timeSpent: 0
        }
      ],
      
      // Achievements and badges
      achievements: [],
      badges: [],
      
      // User preferences
      preferences: {
        experienceLevel: 'beginner', // beginner, intermediate, advanced
        preferredLanguage: 'javascript', // javascript, python, go, java
        useCaseFocus: 'general', // general, retrieval, recommendation, semantic
        pace: 'normal', // slow, normal, fast
        theme: 'light', // light, dark
        fontSize: 14,
        autoSave: true,
        notifications: true
      },
      
      // Progress history for analytics
      progressHistory: [],
      
      // User code snippets
      savedSnippets: [],
      
      // Quiz and assessment results
      assessments: {},
      
      // Session tracking
      sessionStartTime: null,
      totalTimeSpent: 0,
      
      // Custom scenarios created by user
      customScenarios: [],
      
      // Tutorial feedback
      feedback: [],
      
      // Bookmarking
      bookmarks: [],
      
      // Search history
      searchHistory: []
    };
    
    // Initialize session tracking
    this.sessionStartTime = Date.now();
  }

  // Get initial state
  getInitialState() {
    return { ...this.defaultState };
  }

  // Save state to localStorage
  saveToStorage(state) {
    try {
      const serializedState = JSON.stringify(state);
      localStorage.setItem('jadevectordb_tutorial_state', serializedState);
      return true;
    } catch (error) {
      console.error('Failed to save tutorial state:', error);
      return false;
    }
  }

  // Load state from localStorage
  loadFromStorage() {
    try {
      const serializedState = localStorage.getItem('jadevectordb_tutorial_state');
      if (serializedState === null) {
        return null;
      }
      return JSON.parse(serializedState);
    } catch (error) {
      console.error('Failed to load tutorial state:', error);
      return null;
    }
  }

  // Clear storage
  clearStorage() {
    try {
      localStorage.removeItem('jadevectordb_tutorial_state');
      return true;
    } catch (error) {
      console.error('Failed to clear tutorial state:', error);
      return false;
    }
  }

  // Update module progress
  updateModuleProgress(moduleId, completedSteps) {
    const state = this.loadFromStorage() || this.getInitialState();
    
    if (moduleId >= 0 && moduleId < state.modules.length) {
      const previousCompleted = state.modules[moduleId].completedSteps;
      state.modules[moduleId].completedSteps = Math.min(completedSteps, state.modules[moduleId].steps);
      
      // If module is now complete, set completion date
      if (state.modules[moduleId].completedSteps === state.modules[moduleId].steps && 
          previousCompleted < state.modules[moduleId].steps) {
        state.modules[moduleId].completionDate = new Date().toISOString();
        
        // Unlock next module if it exists
        if (moduleId + 1 < state.modules.length) {
          state.modules[moduleId + 1].unlocked = true;
        }
        
        // Add to progress history
        state.progressHistory.push({
          type: 'module_completed',
          moduleId: moduleId,
          timestamp: new Date().toISOString()
        });
      }
      
      this.saveToStorage(state);
    }
    
    return state;
  }

  // Update user preferences
  updatePreferences(preferences) {
    const state = this.loadFromStorage() || this.getInitialState();
    state.preferences = { ...state.preferences, ...preferences };
    this.saveToStorage(state);
    return state;
  }

  // Add achievement
  addAchievement(achievement) {
    const state = this.loadFromStorage() || this.getInitialState();
    
    // Prevent duplicate achievements
    const exists = state.achievements.some(a => a.id === achievement.id);
    if (!exists) {
      state.achievements.push(achievement);
      state.progressHistory.push({
        type: 'achievement_unlocked',
        achievement: achievement.id,
        timestamp: new Date().toISOString()
      });
      this.saveToStorage(state);
    }
    
    return state;
  }

  // Add badge
  addBadge(badge) {
    const state = this.loadFromStorage() || this.getInitialState();
    
    // Prevent duplicate badges
    const exists = state.badges.some(b => b.id === badge.id);
    if (!exists) {
      state.badges.push(badge);
      state.progressHistory.push({
        type: 'badge_earned',
        badge: badge.id,
        timestamp: new Date().toISOString()
      });
      this.saveToStorage(state);
    }
    
    return state;
  }

  // Reset all progress
  resetProgress() {
    this.clearStorage();
    return this.getInitialState();
  }

  // Get overall progress percentage
  getProgressPercentage() {
    const state = this.loadFromStorage() || this.getInitialState();
    
    const totalSteps = state.modules.reduce((sum, module) => sum + module.steps, 0);
    const completedSteps = state.modules.reduce((sum, module) => sum + module.completedSteps, 0);
    
    return totalSteps > 0 ? Math.round((completedSteps / totalSteps) * 100) : 0;
  }

  // Get module progress percentage
  getModuleProgressPercentage(moduleId) {
    const state = this.loadFromStorage() || this.getInitialState();
    
    if (moduleId >= 0 && moduleId < state.modules.length) {
      const module = state.modules[moduleId];
      return module.steps > 0 ? Math.round((module.completedSteps / module.steps) * 100) : 0;
    }
    
    return 0;
  }

  // Save code snippet
  saveCodeSnippet(snippet) {
    const state = this.loadFromStorage() || this.getInitialState();
    
    // Add timestamp and unique ID
    const newSnippet = {
      ...snippet,
      id: Date.now().toString(),
      createdAt: new Date().toISOString()
    };
    
    state.savedSnippets.push(newSnippet);
    this.saveToStorage(state);
    
    return state;
  }

  // Delete code snippet
  deleteCodeSnippet(snippetId) {
    const state = this.loadFromStorage() || this.getInitialState();
    state.savedSnippets = state.savedSnippets.filter(snippet => snippet.id !== snippetId);
    this.saveToStorage(state);
    return state;
  }

  // Save assessment result
  saveAssessmentResult(moduleId, step, result) {
    const state = this.loadFromStorage() || this.getInitialState();
    
    if (!state.assessments[moduleId]) {
      state.assessments[moduleId] = {};
    }
    
    state.assessments[moduleId][step] = {
      ...result,
      timestamp: new Date().toISOString()
    };
    
    this.saveToStorage(state);
    return state;
  }

  // Add custom scenario
  addCustomScenario(scenario) {
    const state = this.loadFromStorage() || this.getInitialState();
    
    const newScenario = {
      ...scenario,
      id: Date.now().toString(),
      createdAt: new Date().toISOString()
    };
    
    state.customScenarios.push(newScenario);
    this.saveToStorage(state);
    return state;
  }

  // Add bookmark
  addBookmark(bookmark) {
    const state = this.loadFromStorage() || this.getInitialState();
    
    // Prevent duplicate bookmarks
    const exists = state.bookmarks.some(b => b.url === bookmark.url);
    if (!exists) {
      state.bookmarks.push(bookmark);
      this.saveToStorage(state);
    }
    
    return state;
  }

  // Remove bookmark
  removeBookmark(bookmarkUrl) {
    const state = this.loadFromStorage() || this.getInitialState();
    state.bookmarks = state.bookmarks.filter(b => b.url !== bookmarkUrl);
    this.saveToStorage(state);
    return state;
  }

  // Add search to history
  addSearchToHistory(searchTerm) {
    const state = this.loadFromStorage() || this.getInitialState();
    
    // Add to search history (limit to 50 items)
    state.searchHistory.unshift({
      term: searchTerm,
      timestamp: new Date().toISOString()
    });
    
    if (state.searchHistory.length > 50) {
      state.searchHistory = state.searchHistory.slice(0, 50);
    }
    
    this.saveToStorage(state);
    return state;
  }

  // Add feedback
  addFeedback(feedback) {
    const state = this.loadFromStorage() || this.getInitialState();
    
    const newFeedback = {
      ...feedback,
      id: Date.now().toString(),
      timestamp: new Date().toISOString()
    };
    
    state.feedback.push(newFeedback);
    this.saveToStorage(state);
    return state;
  }

  // Update time spent on current module
  updateTimeSpent(moduleId, seconds) {
    const state = this.loadFromStorage() || this.getInitialState();
    
    if (moduleId >= 0 && moduleId < state.modules.length) {
      state.modules[moduleId].timeSpent += seconds;
      state.totalTimeSpent += seconds;
      this.saveToStorage(state);
    }
    
    return state;
  }

  // Get session duration
  getSessionDuration() {
    return Date.now() - this.sessionStartTime;
  }

  // Export progress data
  exportProgress() {
    const state = this.loadFromStorage() || this.getInitialState();
    return JSON.stringify(state, null, 2);
  }

  // Import progress data
  importProgress(progressData) {
    try {
      const parsedData = JSON.parse(progressData);
      this.saveToStorage(parsedData);
      return true;
    } catch (error) {
      console.error('Failed to import progress data:', error);
      return false;
    }
  }

  // Get statistics for dashboard
  getStatistics() {
    const state = this.loadFromStorage() || this.getInitialState();
    
    return {
      totalProgress: this.getProgressPercentage(),
      modulesCompleted: state.modules.filter(m => m.completedSteps === m.steps).length,
      totalModules: state.modules.length,
      achievementsUnlocked: state.achievements.length,
      badgesEarned: state.badges.length,
      totalTimeSpent: state.totalTimeSpent + this.getSessionDuration(),
      currentStreak: this.calculateCurrentStreak(state),
      longestStreak: this.calculateLongestStreak(state)
    };
  }

  // Calculate current streak (consecutive days with activity)
  calculateCurrentStreak(state) {
    if (state.progressHistory.length === 0) return 0;
    
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    
    let streak = 0;
    let currentDate = today;
    
    // Sort history by timestamp (newest first)
    const sortedHistory = [...state.progressHistory].sort((a, b) => 
      new Date(b.timestamp) - new Date(a.timestamp)
    );
    
    for (const entry of sortedHistory) {
      const entryDate = new Date(entry.timestamp);
      entryDate.setHours(0, 0, 0, 0);
      
      // If entry is from today or yesterday and we have activity
      const diffDays = Math.floor((currentDate - entryDate) / (1000 * 60 * 60 * 24));
      
      if (diffDays === 0 || diffDays === 1) {
        streak++;
        currentDate = entryDate;
      } else if (diffDays > 1) {
        // Break in streak
        break;
      }
    }
    
    return streak;
  }

  // Calculate longest streak
  calculateLongestStreak(state) {
    // This would require more complex tracking over time
    // For now, we'll return a placeholder
    return state.progressHistory.length > 0 ? Math.min(7, state.progressHistory.length) : 0;
  }
}

export default new AdvancedTutorialState();