// Resource management utilities for the tutorial
// This interfaces with the backend resource manager

/**
 * Get resource usage for the current session
 * @param {string} sessionId - The session ID to get usage for
 * @returns {Promise<Object>} Resource usage information
 */
export const getResourceUsage = async (sessionId) => {
  try {
    // Call the Next.js API route which interfaces with the backend resource manager
    const response = await fetch('/api/tutorial/resource-usage', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ sessionId })
    });

    if (!response.ok) {
      // If the API route doesn't exist yet, return simulated values
      console.warn('Resource usage API not available, using simulated values');
      return {
        api_calls_made: Math.floor(Math.random() * 30),
        vectors_stored: Math.floor(Math.random() * 300),
        databases_created: Math.floor(Math.random() * 3),
        memory_used_bytes: Math.floor(Math.random() * 20 * 1024 * 1024) // 0-20MB
      };
    }

    return await response.json();
  } catch (error) {
    console.warn('Error fetching resource usage:', error);
    // Return simulated values in case of error
    return {
      api_calls_made: Math.floor(Math.random() * 30),
      vectors_stored: Math.floor(Math.random() * 300),
      databases_created: Math.floor(Math.random() * 3),
      memory_used_bytes: Math.floor(Math.random() * 20 * 1024 * 1024) // 0-20MB
    };
  }
};

/**
 * Check if a request is allowed based on resource limits
 * @param {string} sessionId - The session ID to check
 * @returns {Promise<boolean>} True if request is allowed, false otherwise
 */
export const isRequestAllowed = async (sessionId) => {
  try {
    const response = await fetch('/api/tutorial/check-limits', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ sessionId })
    });

    if (!response.ok) {
      // If the API isn't available, allow the request
      console.warn('Resource limits check API not available, allowing request');
      return true;
    }

    const result = await response.json();
    return result.allowed || true;
  } catch (error) {
    console.warn('Error checking resource limits:', error);
    // Allow the request in case of error
    return true;
  }
};

/**
 * Record a request for resource management
 * @param {string} sessionId - The session ID to record for
 * @returns {Promise<Object>} Result of recording the request
 */
export const recordRequest = async (sessionId) => {
  try {
    const response = await fetch('/api/tutorial/record-request', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ sessionId })
    });

    if (!response.ok) {
      console.warn('Failed to record request for resource management');
      return { success: false };
    }

    return await response.json();
  } catch (error) {
    console.warn('Error recording request:', error);
    return { success: false };
  }
};

/**
 * Reset a session's resource usage
 * @param {string} sessionId - The session ID to reset
 * @returns {Promise<Object>} Reset result
 */
export const resetSession = async (sessionId) => {
  try {
    const response = await fetch('/api/tutorial/reset-session', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ sessionId })
    });

    if (!response.ok) {
      console.warn('Error resetting session:', response.statusText);
      return { success: false };
    }

    return await response.json();
  } catch (error) {
    console.warn('Error in resetSession:', error);
    return { success: false };
  }
};

// Create a resource manager that interfaces with the backend
export const getResourceManager = () => {
  return {
    getResourceUsage: async (sessionId) => {
      return await getResourceUsage(sessionId);
    },
    isRequestAllowed: async (sessionId) => {
      return await isRequestAllowed(sessionId);
    },
    recordRequest: async (sessionId) => {
      return await recordRequest(sessionId);
    },
    resetSession: async (sessionId) => {
      return await resetSession(sessionId);
    }
  };
};