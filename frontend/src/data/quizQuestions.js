/**
 * Quiz Questions for JadeVectorDB Tutorials
 * Organized by module with multiple question types
 */

export const quizQuestions = {
  // CLI Basics Module Quiz
  'cli-basics': {
    id: 'cli-basics',
    title: 'CLI Basics Quiz',
    description: 'Test your understanding of JadeVectorDB CLI fundamentals',
    timeLimit: 600, // 10 minutes
    passingScore: 70,
    questions: [
      {
        id: 'cb-q1',
        type: 'multiple-choice',
        question: 'Which command is used to create a new database in JadeVectorDB CLI?',
        options: [
          'jade-db new-database',
          'jade-db create-db',
          'jade-db database create',
          'jade-db init-db'
        ],
        correctAnswer: 1,
        points: 10,
        explanation: 'The correct command is `jade-db create-db` which creates a new vector database with specified parameters.'
      },
      {
        id: 'cb-q2',
        type: 'multiple-choice',
        question: 'What does the `--dimension` parameter specify when creating a database?',
        options: [
          'The number of vectors to store',
          'The size of each vector (number of dimensions)',
          'The database storage size',
          'The number of indexes'
        ],
        correctAnswer: 1,
        points: 10,
        explanation: 'The `--dimension` parameter specifies the vector dimension, which is the number of elements in each vector (e.g., 128, 384, 768).'
      },
      {
        id: 'cb-q3',
        type: 'code-completion',
        question: 'Complete the command to store a vector with ID "vec001" and 3 dimensions [0.1, 0.2, 0.3] in database "test_db":',
        placeholder: 'jade-db --database test_db store-vector ...',
        correctAnswer: 'jade-db --database test_db store-vector --id vec001 --vector "[0.1,0.2,0.3]"',
        points: 15,
        explanation: 'The complete command requires the database name, vector ID, and vector data in JSON array format.'
      },
      {
        id: 'cb-q4',
        type: 'multiple-choice',
        question: 'Which environment variable can be used to set the default JadeVectorDB API URL?',
        options: [
          'JADE_DB_URL',
          'JADEVECTORDB_URL',
          'VECTOR_DB_URL',
          'DATABASE_API_URL'
        ],
        correctAnswer: 1,
        points: 10,
        explanation: 'JADEVECTORDB_URL is the correct environment variable for setting the default API endpoint.'
      },
      {
        id: 'cb-q5',
        type: 'multiple-choice',
        question: 'What is the purpose of the `--api-key` parameter?',
        options: [
          'To encrypt the database',
          'To authenticate requests to the API',
          'To generate a new API key',
          'To list available keys'
        ],
        correctAnswer: 1,
        points: 10,
        explanation: 'The `--api-key` parameter provides authentication credentials for API requests.'
      },
      {
        id: 'cb-q6',
        type: 'scenario-based',
        question: 'You want to search for the 5 most similar vectors to a query vector [0.5, 0.5, 0.5]. Which parameter should you use to limit the results?',
        options: [
          '--limit 5',
          '--top-k 5',
          '--max-results 5',
          '--count 5'
        ],
        correctAnswer: 1,
        points: 15,
        explanation: 'The `--top-k` parameter specifies the number of nearest neighbors to return in similarity search.'
      },
      {
        id: 'cb-q7',
        type: 'multiple-choice',
        question: 'What index type provides the best performance for approximate nearest neighbor search?',
        options: [
          'FLAT',
          'HNSW',
          'IVF',
          'LSH'
        ],
        correctAnswer: 1,
        points: 10,
        explanation: 'HNSW (Hierarchical Navigable Small World) provides excellent performance for ANN search with good recall.'
      },
      {
        id: 'cb-q8',
        type: 'debugging',
        question: 'Fix this command that\'s trying to create a database:\n```\njade-db create-db --name mydb --dimensions 128\n```',
        correctAnswer: '--dimension',
        points: 20,
        explanation: 'The parameter should be `--dimension` (singular), not `--dimensions` (plural).'
      }
    ]
  },

  // CLI Advanced Module Quiz
  'cli-advanced': {
    id: 'cli-advanced',
    title: 'Advanced CLI Operations Quiz',
    description: 'Test your knowledge of advanced CLI features and batch operations',
    timeLimit: 900, // 15 minutes
    passingScore: 70,
    questions: [
      {
        id: 'ca-q1',
        type: 'multiple-choice',
        question: 'What is the recommended batch size for optimal performance when importing vectors?',
        options: [
          '10-50 vectors per batch',
          '50-100 vectors per batch',
          '100-500 vectors per batch',
          '1000+ vectors per batch'
        ],
        correctAnswer: 2,
        points: 10,
        explanation: 'Batches of 100-500 vectors provide a good balance between throughput and memory usage.'
      },
      {
        id: 'ca-q2',
        type: 'code-completion',
        question: 'Complete the command to perform a filtered search for vectors where category="tech":',
        placeholder: 'jade-db --database mydb search --vector "[0.1,0.2,0.3]" ...',
        correctAnswer: 'jade-db --database mydb search --vector "[0.1,0.2,0.3]" --filter \'{"category":"tech"}\'',
        points: 20,
        explanation: 'Metadata filters are passed as JSON objects using the `--filter` parameter.'
      },
      {
        id: 'ca-q3',
        type: 'multiple-choice',
        question: 'Which similarity metric is best for normalized vectors (unit length)?',
        options: [
          'Euclidean distance',
          'Cosine similarity',
          'Manhattan distance',
          'Hamming distance'
        ],
        correctAnswer: 1,
        points: 15,
        explanation: 'For normalized vectors, cosine similarity is computationally efficient and gives identical rankings to Euclidean distance.'
      },
      {
        id: 'ca-q4',
        type: 'scenario-based',
        question: 'You need to update metadata for an existing vector without changing its vector data. What command should you use?',
        options: [
          'jade-db update-vector --id vec001 --metadata \'{"key":"value"}\'',
          'jade-db store-vector --id vec001 --metadata \'{"key":"value"}\'',
          'jade-db modify-metadata --id vec001 --data \'{"key":"value"}\'',
          'jade-db set-metadata --vector-id vec001 \'{"key":"value"}\''
        ],
        correctAnswer: 0,
        points: 15,
        explanation: 'The `update-vector` command allows updating metadata without changing the vector data.'
      },
      {
        id: 'ca-q5',
        type: 'multiple-choice',
        question: 'What happens when you store a vector with an ID that already exists?',
        options: [
          'An error is returned',
          'The vector is appended',
          'The existing vector is updated (upsert)',
          'A new vector is created with a different ID'
        ],
        correctAnswer: 2,
        points: 10,
        explanation: 'JadeVectorDB performs an upsert operation: if the ID exists, it updates the vector; otherwise, it creates a new one.'
      },
      {
        id: 'ca-q6',
        type: 'debugging',
        question: 'This batch import script is failing. What\'s wrong?\n```python\nfor batch in batches:\n    client.store_vectors(db_name, batch)\n    time.sleep(0.1)\n```',
        correctAnswer: 'batch_store_vectors',
        points: 20,
        explanation: 'The method should be `batch_store_vectors` or `store_vectors_batch`, not `store_vectors` for batch operations.'
      },
      {
        id: 'ca-q7',
        type: 'multiple-choice',
        question: 'What is the purpose of the `--include-metadata` flag in search queries?',
        options: [
          'To filter by metadata',
          'To return metadata fields in the response',
          'To sort results by metadata values',
          'To validate metadata format'
        ],
        correctAnswer: 1,
        points: 10,
        explanation: 'The `--include-metadata` flag ensures that vector metadata is included in the search results.'
      },
      {
        id: 'ca-q8',
        type: 'code-completion',
        question: 'Write a filter to find vectors where score > 0.8 AND category is either "tech" or "science":',
        placeholder: '{"$and": [...]}',
        correctAnswer: '{"$and":[{"score":{"$gt":0.8}},{"category":{"$in":["tech","science"]}}]}',
        points: 25,
        explanation: 'Complex filters use MongoDB-style query operators like $and, $gt, and $in.'
      }
    ]
  },

  // Vector Fundamentals Quiz
  'vector-fundamentals': {
    id: 'vector-fundamentals',
    title: 'Vector Database Fundamentals Quiz',
    description: 'Test your understanding of vector database concepts',
    timeLimit: 600, // 10 minutes
    passingScore: 70,
    questions: [
      {
        id: 'vf-q1',
        type: 'multiple-choice',
        question: 'What is a vector embedding?',
        options: [
          'A compressed file format',
          'A numerical representation of data in high-dimensional space',
          'A database index structure',
          'An encryption algorithm'
        ],
        correctAnswer: 1,
        points: 10,
        explanation: 'Vector embeddings are numerical representations that capture semantic meaning in high-dimensional space.'
      },
      {
        id: 'vf-q2',
        type: 'multiple-choice',
        question: 'What does "dimension" mean in vector databases?',
        options: [
          'The size of the database',
          'The number of vectors stored',
          'The number of values in each vector',
          'The number of metadata fields'
        ],
        correctAnswer: 2,
        points: 10,
        explanation: 'Dimension refers to the number of elements/values in each vector (e.g., a 384-dimensional vector has 384 numbers).'
      },
      {
        id: 'vf-q3',
        type: 'multiple-choice',
        question: 'What is the primary use case for vector databases?',
        options: [
          'Storing relational data',
          'Semantic/similarity search',
          'Transaction processing',
          'Time-series analysis'
        ],
        correctAnswer: 1,
        points: 15,
        explanation: 'Vector databases excel at semantic and similarity search, finding items similar to a query based on meaning rather than exact matches.'
      },
      {
        id: 'vf-q4',
        type: 'multiple-choice',
        question: 'Which of these is NOT a common distance metric for vector similarity?',
        options: [
          'Cosine similarity',
          'Euclidean distance',
          'Hamming distance',
          'SQL distance'
        ],
        correctAnswer: 3,
        points: 10,
        explanation: 'SQL distance is not a real metric. Common metrics include cosine similarity, Euclidean distance, and Manhattan distance.'
      },
      {
        id: 'vf-q5',
        type: 'scenario-based',
        question: 'You have text documents and want to find semantically similar ones. What should you do first?',
        options: [
          'Store the raw text directly',
          'Convert text to embeddings using a model',
          'Create keyword indexes',
          'Split text into sentences'
        ],
        correctAnswer: 1,
        points: 15,
        explanation: 'Text must be converted to vector embeddings using models like BERT, GPT, or sentence transformers before storage.'
      },
      {
        id: 'vf-q6',
        type: 'multiple-choice',
        question: 'What is the trade-off with approximate nearest neighbor (ANN) algorithms?',
        options: [
          'Storage vs. retrieval speed',
          'Accuracy vs. speed',
          'Security vs. performance',
          'Scalability vs. cost'
        ],
        correctAnswer: 1,
        points: 15,
        explanation: 'ANN algorithms trade perfect accuracy for much faster search times, typically achieving 95%+ recall.'
      },
      {
        id: 'vf-q7',
        type: 'multiple-choice',
        question: 'Why is metadata important in vector databases?',
        options: [
          'It increases vector dimensions',
          'It improves embedding quality',
          'It enables filtering and provides context',
          'It speeds up similarity calculations'
        ],
        correctAnswer: 2,
        points: 10,
        explanation: 'Metadata allows filtering results and provides business context (e.g., category, timestamp, source) without affecting similarity search.'
      },
      {
        id: 'vf-q8',
        type: 'multiple-choice',
        question: 'What is "top-k" in similarity search?',
        options: [
          'The highest scoring vector',
          'The k most similar vectors to the query',
          'The k-means clustering result',
          'The top k databases'
        ],
        correctAnswer: 1,
        points: 15,
        explanation: 'Top-k search returns the k nearest neighbors (most similar vectors) to the query vector.'
      }
    ]
  },

  // API Integration Quiz
  'api-integration': {
    id: 'api-integration',
    title: 'API Integration Quiz',
    description: 'Test your knowledge of JadeVectorDB API usage and integration',
    timeLimit: 900, // 15 minutes
    passingScore: 70,
    questions: [
      {
        id: 'api-q1',
        type: 'multiple-choice',
        question: 'What is the base URL pattern for JadeVectorDB API endpoints?',
        options: [
          'http://localhost:8080/api/',
          'http://localhost:8080/v1/',
          'http://localhost:8080/jade/',
          'http://localhost:8080/vectordb/'
        ],
        correctAnswer: 1,
        points: 10,
        explanation: 'JadeVectorDB API endpoints follow the pattern `/v1/` for version 1 of the API.'
      },
      {
        id: 'api-q2',
        type: 'code-completion',
        question: 'Complete this cURL command to create a database:\n```bash\ncurl -X POST http://localhost:8080/v1/databases \\\n  -H "Content-Type: application/json" \\\n  -H "X-API-Key: your-key" \\\n  -d \'...\'\n```',
        correctAnswer: '{"name":"mydb","description":"Test database","dimension":384,"indexType":"HNSW"}',
        points: 20,
        explanation: 'The request body must include name, dimension, and optionally description and indexType.'
      },
      {
        id: 'api-q3',
        type: 'multiple-choice',
        question: 'Which HTTP header is required for API authentication?',
        options: [
          'Authorization: Bearer token',
          'X-API-Key: key',
          'Auth-Token: token',
          'API-Secret: secret'
        ],
        correctAnswer: 1,
        points: 10,
        explanation: 'JadeVectorDB uses the `X-API-Key` header for API authentication.'
      },
      {
        id: 'api-q4',
        type: 'multiple-choice',
        question: 'What HTTP method is used to update a vector?',
        options: [
          'POST',
          'PATCH',
          'PUT',
          'UPDATE'
        ],
        correctAnswer: 2,
        points: 10,
        explanation: 'PUT is used for updating resources, including vector data and metadata.'
      },
      {
        id: 'api-q5',
        type: 'scenario-based',
        question: 'You receive a 401 Unauthorized error. What is the most likely cause?',
        options: [
          'Invalid vector dimension',
          'Missing or invalid API key',
          'Database doesn\'t exist',
          'Server is down'
        ],
        correctAnswer: 1,
        points: 15,
        explanation: 'A 401 error indicates authentication failure, usually due to missing or invalid API key.'
      },
      {
        id: 'api-q6',
        type: 'code-completion',
        question: 'Complete this Python code to search for vectors:\n```python\nresponse = requests.post(\n    f"{base_url}/databases/{db_id}/search",\n    headers={"X-API-Key": api_key},\n    json={...}\n)\n```',
        correctAnswer: '{"vector":[0.1,0.2,0.3],"topK":5,"includeMetadata":true}',
        points: 20,
        explanation: 'The search request requires a query vector, topK parameter, and optionally includeMetadata flag.'
      },
      {
        id: 'api-q7',
        type: 'debugging',
        question: 'This request is failing with 400 Bad Request:\n```json\n{\n  "vectors": [\n    {"id": "v1", "data": [0.1, 0.2]}\n  ]\n}\n```\nWhat\'s missing?',
        correctAnswer: 'vector',
        points: 20,
        explanation: 'The field should be named "vector" not "data" in the API request.'
      },
      {
        id: 'api-q8',
        type: 'multiple-choice',
        question: 'What does the `includeVectorData` parameter do in search responses?',
        options: [
          'Includes the query vector in results',
          'Includes the matched vectors\' data in results',
          'Enables vector validation',
          'Includes vector statistics'
        ],
        correctAnswer: 1,
        points: 15,
        explanation: '`includeVectorData` controls whether the full vector arrays are returned in search results (useful for reducing response size).'
      },
      {
        id: 'api-q9',
        type: 'multiple-choice',
        question: 'Which endpoint would you use to delete a specific vector?',
        options: [
          'DELETE /v1/vectors/{id}',
          'DELETE /v1/databases/{db_id}/vectors/{vector_id}',
          'POST /v1/databases/{db_id}/vectors/delete',
          'PUT /v1/databases/{db_id}/vectors/{vector_id}/remove'
        ],
        correctAnswer: 1,
        points: 15,
        explanation: 'The correct endpoint pattern is DELETE /v1/databases/{db_id}/vectors/{vector_id}.'
      }
    ]
  }
};

// Helper function to get quiz by ID
export function getQuiz(quizId) {
  return quizQuestions[quizId] || null;
}

// Helper function to get all available quizzes
export function getAllQuizzes() {
  return Object.values(quizQuestions);
}

// Helper function to get quiz titles for navigation
export function getQuizTitles() {
  return Object.entries(quizQuestions).map(([id, quiz]) => ({
    id,
    title: quiz.title,
    description: quiz.description
  }));
}

export default quizQuestions;
