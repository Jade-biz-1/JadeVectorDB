/**
 * Authentication service for the tutorial sessions
 * Generates temporary API keys for each tutorial session
 */

class AuthService {
  constructor() {
    this.storageKey = 'jadevectordb-tutorial-auth';
  }

  /**
   * Generate a temporary API key for the tutorial session
   * @returns {string} A temporary API key
   */
  generateTemporaryApiKey() {
    // Create a temporary API key that's valid for the session
    const apiKey = 'tutorial_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    localStorage.setItem(this.storageKey, apiKey);
    return apiKey;
  }

  /**
   * Get the current API key for the session
   * @returns {string|null} The current API key or null if not found
   */
  getCurrentApiKey() {
    return localStorage.getItem(this.storageKey);
  }

  /**
   * Remove the current API key (logout)
   */
  removeApiKey() {
    localStorage.removeItem(this.storageKey);
  }

  /**
   * Get authentication headers for API requests
   * @returns {Object} Headers object with authentication info
   */
  getAuthHeaders() {
    const apiKey = this.getCurrentApiKey();
    if (!apiKey) {
      // Generate a temporary key if one doesn't exist
      return { 'Authorization': `Bearer ${this.generateTemporaryApiKey()}` };
    }
    return { 'Authorization': `Bearer ${apiKey}` };
  }
}

// Create a singleton instance
let authServiceInstance = null;

export const getAuthService = () => {
  if (!authServiceInstance) {
    authServiceInstance = new AuthService();
  }
  return authServiceInstance;
};

// Initialize with a temporary key if one doesn't exist
if (typeof window !== 'undefined') {
  const authService = getAuthService();
  if (!authService.getCurrentApiKey()) {
    authService.generateTemporaryApiKey();
  }
}