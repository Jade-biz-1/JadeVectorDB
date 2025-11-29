import { getResourceManager } from '../lib/resourceManager';
import { getAuthService } from './auth';

// Base URL for the backend API
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8080';

/**
 * Service class for all JadeVectorDB API operations
 */
class ApiService {
  constructor() {
    this.sessionId = this.generateSessionId();
    this.resourceManager = getResourceManager();
    this.authService = getAuthService();
  }

  /**
   * Generate a unique session ID for the tutorial session
   * @returns {string} A unique session identifier
   */
  generateSessionId() {
    if (typeof window !== 'undefined') {
      // Try to get existing session from localStorage
      const existingSession = localStorage.getItem('jadevectordb-tutorial-session');
      if (existingSession) {
        return existingSession;
      }
      
      // Generate a new session ID
      const sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
      localStorage.setItem('jadevectordb-tutorial-session', sessionId);
      return sessionId;
    }
    return 'session_server_' + Date.now();
  }

  /**
   * Check if request is allowed based on resource limits
   * @returns {boolean} True if request is allowed, false otherwise
   */
  async checkResourceLimits() {
    // Use the resource manager to check limits
    return await this.resourceManager.isRequestAllowed(this.sessionId);
  }

  /**
   * Record the API request for resource management
   */
  async recordRequest() {
    // Use the resource manager to record the request
    await this.resourceManager.recordRequest(this.sessionId);
  }

  /**
   * Make a request to the backend API with proper error handling
   * @param {string} endpoint - The API endpoint to call
   * @param {string} method - HTTP method (GET, POST, PUT, DELETE)
   * @param {Object} body - Request body for POST/PUT requests
   * @param {Object} headers - Additional headers to send
   * @returns {Promise<Object>} The response from the API
   */
  async makeRequest(endpoint, method = 'GET', body = null, headers = {}) {
    // Check if request is allowed based on resource limits
    if (!(await this.checkResourceLimits())) {
      throw new Error('Resource limits exceeded. Please try again later.');
    }

    // Record the request for resource management
    await this.recordRequest();

    // Construct the full URL
    const url = endpoint.startsWith('http') ? endpoint : `${API_BASE_URL}${endpoint}`;

    // Get authentication headers
    const authHeaders = this.authService.getAuthHeaders();

    // Prepare request options
    const options = {
      method,
      headers: {
        'Content-Type': 'application/json',
        'X-Session-ID': this.sessionId,
        ...authHeaders,
        ...headers
      }
    };

    // Add body for methods that support it
    if (body && (method === 'POST' || method === 'PUT' || method === 'PATCH')) {
      options.body = JSON.stringify(body);
    }

    try {
      // Make the request to the backend
      const response = await fetch(url, options);

      // Record the request for resource management after successful call
      await this.recordRequest();

      // Check if the response is ok (status 200-299)
      if (!response.ok) {
        // Try to get error details from the response
        let errorDetails = `HTTP error! status: ${response.status}`;
        let errorData = null;
        
        try {
          errorData = await response.json();
          errorDetails = errorData.message || errorData.error || errorData.detail || errorDetails;
        } catch (parseError) {
          // If we can't parse the error response as JSON, try text
          try {
            const errorText = await response.text();
            if (errorText) {
              errorDetails = errorText;
            } else {
              // If we still can't get details, use status text
              errorDetails = response.statusText || errorDetails;
            }
          } catch (textError) {
            // If all parsing fails, use status text
            errorDetails = response.statusText || errorDetails;
          }
        }
        
        throw new Error(errorDetails);
      }

      // Return the response JSON
      return await response.json();
    } catch (error) {
      // Handle different types of errors appropriately
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        // Network error
        throw new Error('Network error: Could not connect to the server. Please check your connection and try again.');
      } else if (error.message.includes('Resource limits exceeded')) {
        // Re-throw resource limit errors
        throw error;
      } else if (error.message.includes('HTTP error')) {
        // Already formatted HTTP error
        throw error;
      } else {
        // Other errors
        throw new Error(`Request failed: ${error.message}`);
      }
    }
  }

  // Database operations
  async createDatabase(dbConfig) {
    try {
      const response = await this.makeRequest('/v1/databases', 'POST', dbConfig);
      return response;
    } catch (error) {
      console.error('Error creating database:', error);
      throw new Error(`Failed to create database: ${error.message}`);
    }
  }

  async listDatabases() {
    try {
      const response = await this.makeRequest('/v1/databases', 'GET');
      return response.databases || [];
    } catch (error) {
      console.error('Error listing databases:', error);
      throw new Error(`Failed to list databases: ${error.message}`);
    }
  }

  async getDatabase(databaseId) {
    try {
      const response = await this.makeRequest(`/v1/databases/${databaseId}`, 'GET');
      return response;
    } catch (error) {
      console.error(`Error getting database ${databaseId}:`, error);
      throw new Error(`Failed to get database: ${error.message}`);
    }
  }

  async updateDatabase(databaseId, dbConfig) {
    try {
      const response = await this.makeRequest(`/v1/databases/${databaseId}`, 'PUT', dbConfig);
      return response;
    } catch (error) {
      console.error(`Error updating database ${databaseId}:`, error);
      throw new Error(`Failed to update database: ${error.message}`);
    }
  }

  async deleteDatabase(databaseId) {
    try {
      const response = await this.makeRequest(`/v1/databases/${databaseId}`, 'DELETE');
      return response;
    } catch (error) {
      console.error(`Error deleting database ${databaseId}:`, error);
      throw new Error(`Failed to delete database: ${error.message}`);
    }
  }

  // Vector operations
  async storeVector(databaseId, vector) {
    try {
      const response = await this.makeRequest(`/v1/databases/${databaseId}/vectors`, 'POST', vector);
      return response;
    } catch (error) {
      console.error(`Error storing vector in database ${databaseId}:`, error);
      throw new Error(`Failed to store vector: ${error.message}`);
    }
  }

  async getVector(databaseId, vectorId) {
    try {
      const response = await this.makeRequest(`/v1/databases/${databaseId}/vectors/${vectorId}`, 'GET');
      return response;
    } catch (error) {
      console.error(`Error getting vector ${vectorId} from database ${databaseId}:`, error);
      throw new Error(`Failed to get vector: ${error.message}`);
    }
  }

  async updateVector(databaseId, vectorId, vector) {
    try {
      const response = await this.makeRequest(`/v1/databases/${databaseId}/vectors/${vectorId}`, 'PUT', vector);
      return response;
    } catch (error) {
      console.error(`Error updating vector ${vectorId} in database ${databaseId}:`, error);
      throw new Error(`Failed to update vector: ${error.message}`);
    }
  }

  async deleteVector(databaseId, vectorId) {
    try {
      const response = await this.makeRequest(`/v1/databases/${databaseId}/vectors/${vectorId}`, 'DELETE');
      return response;
    } catch (error) {
      console.error(`Error deleting vector ${vectorId} from database ${databaseId}:`, error);
      throw new Error(`Failed to delete vector: ${error.message}`);
    }
  }

  async batchStoreVectors(databaseId, vectors) {
    try {
      const response = await this.makeRequest(`/v1/databases/${databaseId}/vectors/batch`, 'POST', { vectors });
      return response;
    } catch (error) {
      console.error(`Error batch storing vectors in database ${databaseId}:`, error);
      throw new Error(`Failed to batch store vectors: ${error.message}`);
    }
  }

  // Search operations
  async similaritySearch(databaseId, queryVector, searchParams = {}) {
    try {
      const requestBody = {
        queryVector: queryVector.values || queryVector,
        ...searchParams
      };
      
      const response = await this.makeRequest(`/v1/databases/${databaseId}/search`, 'POST', requestBody);
      return response;
    } catch (error) {
      console.error(`Error performing similarity search in database ${databaseId}:`, error);
      throw new Error(`Failed to perform similarity search: ${error.message}`);
    }
  }

  async advancedSearch(databaseId, queryVector, searchParams = {}) {
    try {
      const requestBody = {
        queryVector: queryVector.values || queryVector,
        ...searchParams
      };
      
      const response = await this.makeRequest(`/v1/databases/${databaseId}/search/advanced`, 'POST', requestBody);
      return response;
    } catch (error) {
      console.error(`Error performing advanced search in database ${databaseId}:`, error);
      throw new Error(`Failed to perform advanced search: ${error.message}`);
    }
  }

  // Index operations
  async createIndex(databaseId, indexConfig) {
    try {
      const response = await this.makeRequest(`/v1/databases/${databaseId}/indexes`, 'POST', indexConfig);
      return response;
    } catch (error) {
      console.error(`Error creating index in database ${databaseId}:`, error);
      throw new Error(`Failed to create index: ${error.message}`);
    }
  }

  async listIndexes(databaseId) {
    try {
      const response = await this.makeRequest(`/v1/databases/${databaseId}/indexes`, 'GET');
      return response.indexes || [];
    } catch (error) {
      console.error(`Error listing indexes in database ${databaseId}:`, error);
      throw new Error(`Failed to list indexes: ${error.message}`);
    }
  }

  async updateIndex(databaseId, indexId, indexConfig) {
    try {
      const response = await this.makeRequest(`/v1/databases/${databaseId}/indexes/${indexId}`, 'PUT', indexConfig);
      return response;
    } catch (error) {
      console.error(`Error updating index ${indexId} in database ${databaseId}:`, error);
      throw new Error(`Failed to update index: ${error.message}`);
    }
  }

  async deleteIndex(databaseId, indexId) {
    try {
      const response = await this.makeRequest(`/v1/databases/${databaseId}/indexes/${indexId}`, 'DELETE');
      return response;
    } catch (error) {
      console.error(`Error deleting index ${indexId} from database ${databaseId}:`, error);
      throw new Error(`Failed to delete index: ${error.message}`);
    }
  }

  // Embedding operations
  async generateEmbedding(text) {
    try {
      const response = await this.makeRequest('/v1/embeddings/generate', 'POST', { text });
      return response;
    } catch (error) {
      console.error('Error generating embedding:', error);
      throw new Error(`Failed to generate embedding: ${error.message}`);
    }
  }

  // Get session info for the resource monitor
  async getSessionInfo() {
    try {
      // Get resource usage from the resource manager
      const resourceUsage = await this.resourceManager.getResourceUsage(this.sessionId);
      
      return {
        sessionId: this.sessionId,
        createdAt: new Date().toISOString(),
        resourceUsage: resourceUsage
      };
    } catch (error) {
      console.error('Error getting session info:', error);
      throw new Error(`Failed to get session info: ${error.message}`);
    }
  }
}

// Create a singleton instance of the API service
let apiServiceInstance = null;

export const getApiService = () => {
  if (!apiServiceInstance) {
    apiServiceInstance = new ApiService();
  }
  return apiServiceInstance;
};

// Export the base URL for use in components that need it
export { API_BASE_URL };