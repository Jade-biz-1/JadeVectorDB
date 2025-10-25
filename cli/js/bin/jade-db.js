#!/usr/bin/env node

const { program } = require('commander');
const chalk = require('chalk');
const { 
  createDatabase, 
  listDatabases, 
  getDatabase, 
  storeVector, 
  retrieveVector, 
  deleteVector,
  searchVectors,
  getHealth,
  getStatus
} = require('../src/api');

// Initialize commander program
program
  .name('jade-db')
  .description('CLI for interacting with JadeVectorDB')
  .version('1.0.0');

// Configuration options
let baseUrl = 'http://localhost:8080';
let apiKey = '';

// Global options
program
  .option('--url <url>', 'JadeVectorDB API URL', 'http://localhost:8080')
  .option('--api-key <key>', 'API key for authentication');

// Database commands
const databaseCommands = program
  .command('database')
  .description('Database management commands');

databaseCommands
  .command('create')
  .description('Create a new database')
  .option('--name <name>', 'Database name (required)')
  .option('--description <description>', 'Database description', '')
  .option('--dimension <dimension>', 'Vector dimension', '128')
  .option('--index-type <type>', 'Index type', 'HNSW')
  .action(async (options) => {
    if (!options.name) {
      console.error(chalk.red('Error: Database name is required'));
      process.exit(1);
    }
    
    program.opts().url && (baseUrl = program.opts().url);
    program.opts().apiKey && (apiKey = program.opts().apiKey);
    
    try {
      const result = await createDatabase(
        baseUrl, 
        apiKey, 
        options.name, 
        options.description, 
        parseInt(options.dimension), 
        options.indexType
      );
      console.log(chalk.green('Database created successfully:'));
      console.log(JSON.stringify(result, null, 2));
    } catch (error) {
      console.error(chalk.red(`Error creating database: ${error.message}`));
      process.exit(1);
    }
  });

databaseCommands
  .command('list')
  .description('List all databases')
  .action(async () => {
    program.opts().url && (baseUrl = program.opts().url);
    program.opts().apiKey && (apiKey = program.opts().apiKey);
    
    try {
      const result = await listDatabases(baseUrl, apiKey);
      console.log(chalk.green('Databases:'));
      console.log(JSON.stringify(result, null, 2));
    } catch (error) {
      console.error(chalk.red(`Error listing databases: ${error.message}`));
      process.exit(1);
    }
  });

databaseCommands
  .command('get')
  .description('Get database details')
  .argument('<id>', 'Database ID')
  .action(async (id) => {
    program.opts().url && (baseUrl = program.opts().url);
    program.opts().apiKey && (apiKey = program.opts().apiKey);
    
    try {
      const result = await getDatabase(baseUrl, apiKey, id);
      console.log(chalk.green('Database details:'));
      console.log(JSON.stringify(result, null, 2));
    } catch (error) {
      console.error(chalk.red(`Error getting database: ${error.message}`));
      process.exit(1);
    }
  });

// Vector commands
const vectorCommands = program
  .command('vector')
  .description('Vector operations');

vectorCommands
  .command('store')
  .description('Store a vector')
  .option('--database-id <id>', 'Database ID (required)')
  .option('--vector-id <id>', 'Vector ID (required)')
  .option('--values <values>', 'Vector values as JSON array (required)')  // e.g., [0.1, 0.2, 0.3]
  .option('--metadata <metadata>', 'Metadata as JSON string')  // e.g., {"category": "test"}
  .action(async (options) => {
    if (!options.databaseId || !options.vectorId || !options.values) {
      console.error(chalk.red('Error: database-id, vector-id, and values are required'));
      process.exit(1);
    }
    
    program.opts().url && (baseUrl = program.opts().url);
    program.opts().apiKey && (apiKey = program.opts().apiKey);
    
    let valuesArray;
    try {
      valuesArray = typeof options.values === 'string' ? JSON.parse(options.values) : options.values;
    } catch (error) {
      console.error(chalk.red(`Error parsing values: ${error.message}`));
      process.exit(1);
    }
    
    let metadataObj = null;
    if (options.metadata) {
      try {
        metadataObj = JSON.parse(options.metadata);
      } catch (error) {
        console.error(chalk.red(`Error parsing metadata: ${error.message}`));
        process.exit(1);
      }
    }
    
    try {
      const result = await storeVector(
        baseUrl, 
        apiKey, 
        options.databaseId, 
        options.vectorId, 
        valuesArray, 
        metadataObj
      );
      console.log(chalk.green('Vector stored successfully:'));
      console.log(JSON.stringify(result, null, 2));
    } catch (error) {
      console.error(chalk.red(`Error storing vector: ${error.message}`));
      process.exit(1);
    }
  });

vectorCommands
  .command('retrieve')
  .description('Retrieve a vector')
  .option('--database-id <id>', 'Database ID (required)')
  .option('--vector-id <id>', 'Vector ID (required)')
  .action(async (options) => {
    if (!options.databaseId || !options.vectorId) {
      console.error(chalk.red('Error: database-id and vector-id are required'));
      process.exit(1);
    }
    
    program.opts().url && (baseUrl = program.opts().url);
    program.opts().apiKey && (apiKey = program.opts().apiKey);
    
    try {
      const result = await retrieveVector(
        baseUrl, 
        apiKey, 
        options.databaseId, 
        options.vectorId
      );
      console.log(chalk.green('Vector retrieved:'));
      console.log(JSON.stringify(result, null, 2));
    } catch (error) {
      console.error(chalk.red(`Error retrieving vector: ${error.message}`));
      process.exit(1);
    }
  });

vectorCommands
  .command('delete')
  .description('Delete a vector')
  .option('--database-id <id>', 'Database ID (required)')
  .option('--vector-id <id>', 'Vector ID (required)')
  .action(async (options) => {
    if (!options.databaseId || !options.vectorId) {
      console.error(chalk.red('Error: database-id and vector-id are required'));
      process.exit(1);
    }
    
    program.opts().url && (baseUrl = program.opts().url);
    program.opts().apiKey && (apiKey = program.opts().apiKey);
    
    try {
      const result = await deleteVector(
        baseUrl, 
        apiKey, 
        options.databaseId, 
        options.vectorId
      );
      console.log(chalk.green('Vector deleted successfully:'));
      console.log(JSON.stringify(result, null, 2));
    } catch (error) {
      console.error(chalk.red(`Error deleting vector: ${error.message}`));
      process.exit(1);
    }
  });

// Search command
program
  .command('search')
  .description('Search for similar vectors')
  .option('--database-id <id>', 'Database ID (required)')
  .option('--query-vector <vector>', 'Query vector as JSON array (required)')
  .option('--top-k <count>', 'Number of results to return', '10')
  .option('--threshold <threshold>', 'Similarity threshold')
  .action(async (options) => {
    if (!options.databaseId || !options.queryVector) {
      console.error(chalk.red('Error: database-id and query-vector are required'));
      process.exit(1);
    }
    
    program.opts().url && (baseUrl = program.opts().url);
    program.opts().apiKey && (apiKey = program.opts().apiKey);
    
    let queryVectorArray;
    try {
      queryVectorArray = typeof options.queryVector === 'string' ? JSON.parse(options.queryVector) : options.queryVector;
    } catch (error) {
      console.error(chalk.red(`Error parsing query vector: ${error.message}`));
      process.exit(1);
    }
    
    const topK = parseInt(options.topK);
    const threshold = options.threshold ? parseFloat(options.threshold) : null;
    
    try {
      const result = await searchVectors(
        baseUrl, 
        apiKey, 
        options.databaseId, 
        queryVectorArray, 
        topK, 
        threshold
      );
      console.log(chalk.green('Search results:'));
      console.log(JSON.stringify(result, null, 2));
    } catch (error) {
      console.error(chalk.red(`Error during search: ${error.message}`));
      process.exit(1);
    }
  });

// Health and status commands
program
  .command('health')
  .description('Get system health status')
  .action(async () => {
    program.opts().url && (baseUrl = program.opts().url);
    program.opts().apiKey && (apiKey = program.opts().apiKey);
    
    try {
      const result = await getHealth(baseUrl, apiKey);
      console.log(chalk.green('Health status:'));
      console.log(JSON.stringify(result, null, 2));
    } catch (error) {
      console.error(chalk.red(`Error getting health status: ${error.message}`));
      process.exit(1);
    }
  });

program
  .command('status')
  .description('Get system status')
  .action(async () => {
    program.opts().url && (baseUrl = program.opts().url);
    program.opts().apiKey && (apiKey = program.opts().apiKey);
    
    try {
      const result = await getStatus(baseUrl, apiKey);
      console.log(chalk.green('System status:'));
      console.log(JSON.stringify(result, null, 2));
    } catch (error) {
      console.error(chalk.red(`Error getting status: ${error.message}`));
      process.exit(1);
    }
  });

// Parse the command line arguments
program.parse();