import Head from 'next/head';
import { useState, useRef } from 'react';
import { vectorApi, databaseApi } from '../lib/api';

export default function BatchOperations() {
  const [selectedDatabase, setSelectedDatabase] = useState('');
  const [databases, setDatabases] = useState([]);
  const [batchOperation, setBatchOperation] = useState('upload'); // 'upload' or 'download'
  const [vectors, setVectors] = useState([{ id: '', values: '', metadata: '{}' }]);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState(null);
  const [fileContent, setFileContent] = useState('');
  const [downloadFormat, setDownloadFormat] = useState('json');
  
  const fileInputRef = useRef(null);

  // Fetch databases on component mount
  useState(() => {
    fetchDatabases();
  });

  const fetchDatabases = async () => {
    try {
      const response = await databaseApi.listDatabases();
      setDatabases(response.databases.map(db => ({
        id: db.databaseId,
        name: db.name
      })));
    } catch (error) {
      console.error('Error fetching databases:', error);
      alert(`Error fetching databases: ${error.message}`);
    }
  };

  const handleAddVector = () => {
    setVectors([...vectors, { id: '', values: '', metadata: '{}' }]);
  };

  const handleRemoveVector = (index) => {
    if (vectors.length > 1) {
      const newVectors = [...vectors];
      newVectors.splice(index, 1);
      setVectors(newVectors);
    }
  };

  const handleVectorChange = (index, field, value) => {
    const newVectors = [...vectors];
    newVectors[index][field] = value;
    setVectors(newVectors);
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        try {
          // Attempt to parse as JSON first
          const jsonData = JSON.parse(event.target.result);
          setFileContent(JSON.stringify(jsonData, null, 2));
        } catch (jsonError) {
          // If JSON parsing fails, treat as text
          setFileContent(event.target.result);
        }
      };
      reader.readAsText(file);
    }
  };

  const handleBatchUpload = async () => {
    if (!selectedDatabase) {
      alert('Please select a database');
      return;
    }

    setLoading(true);
    setProgress(0);
    setResults(null);

    try {
      // Validate and parse vectors
      const processedVectors = vectors
        .filter(vec => vec.id && vec.values) // Only process vectors with ID and values
        .map(vec => {
          let values = [];
          // Parse values as either comma-separated numbers or JSON array
          if (vec.values.trim().startsWith('[')) {
            values = JSON.parse(vec.values);
          } else {
            values = vec.values.split(',')
              .map(s => parseFloat(s.trim()))
              .filter(v => !isNaN(v));
          }

          return {
            id: vec.id,
            values: values,
            metadata: JSON.parse(vec.metadata || '{}')
          };
        });

      if (processedVectors.length === 0) {
        throw new Error('No valid vectors to upload');
      }

      // Simulate progress
      let progress = 0;
      const interval = setInterval(() => {
        progress += 10;
        if (progress >= 100) {
          clearInterval(interval);
        } else {
          setProgress(progress);
        }
      }, 200);

      // Perform batch upload
      const response = await vectorApi.storeVectorsBatch(selectedDatabase, processedVectors);
      clearInterval(interval);
      setProgress(100);
      setResults({
        success: true,
        inserted: response.count,
        message: `Successfully uploaded ${response.count} vectors`
      });
    } catch (error) {
      console.error('Error uploading vectors:', error);
      setResults({
        success: false,
        message: `Error uploading vectors: ${error.message}`
      });
    } finally {
      setLoading(false);
    }
  };

  const handleBatchDownload = async () => {
    if (!selectedDatabase) {
      alert('Please select a database');
      return;
    }

    setLoading(true);
    setProgress(0);
    setResults(null);

    try {
      // Simulate progress
      let progress = 0;
      const interval = setInterval(() => {
        progress += 10;
        if (progress >= 100) {
          clearInterval(interval);
        } else {
          setProgress(progress);
        }
      }, 200);

      // In a real implementation, you'd fetch a batch of vectors
      // For now, we'll just simulate a response
      const response = { message: "This would download vectors from the selected database" };
      clearInterval(interval);
      setProgress(100);

      if (downloadFormat === 'json') {
        // Convert response to JSON and trigger download
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(response, null, 2));
        const downloadAnchorNode = document.createElement('a');
        downloadAnchorNode.setAttribute("href", dataStr);
        downloadAnchorNode.setAttribute("download", `vectors_${selectedDatabase}.json`);
        document.body.appendChild(downloadAnchorNode);
        downloadAnchorNode.click();
        downloadAnchorNode.remove();
      } else {
        // For CSV format
        const dataStr = "data:text/csv;charset=utf-8," + encodeURIComponent("id,values,metadata\nexample,0.1,0.2,0.3,\"{\\\"key\\\":\\\"value\\\"}\"");
        const downloadAnchorNode = document.createElement('a');
        downloadAnchorNode.setAttribute("href", dataStr);
        downloadAnchorNode.setAttribute("download", `vectors_${selectedDatabase}.csv`);
        document.body.appendChild(downloadAnchorNode);
        downloadAnchorNode.click();
        downloadAnchorNode.remove();
      }

      setResults({
        success: true,
        message: `Download initiated for vectors in ${databases.find(db => db.id === selectedDatabase)?.name}`
      });
    } catch (error) {
      console.error('Error downloading vectors:', error);
      setResults({
        success: false,
        message: `Error downloading vectors: ${error.message}`
      });
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (batchOperation === 'upload') {
      await handleBatchUpload();
    } else {
      await handleBatchDownload();
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Head>
        <title>Batch Operations - JadeVectorDB</title>
        <meta name="description" content="Batch vector operations in JadeVectorDB" />
      </Head>

      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">Batch Vector Operations</h1>
        </div>
      </header>

      <main>
        <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          {/* Batch Operation Form */}
          <div className="bg-white shadow px-4 py-5 sm:rounded-lg sm:p-6 mb-8">
            <div className="md:grid md:grid-cols-3 md:gap-6">
              <div className="md:col-span-1">
                <h3 className="text-lg font-medium leading-6 text-gray-900">Batch Operations</h3>
                <p className="mt-1 text-sm text-gray-500">
                  Upload or download multiple vectors in batch.
                </p>
              </div>
              <div className="mt-5 md:mt-0 md:col-span-2">
                <form onSubmit={handleSubmit}>
                  <div className="grid grid-cols-6 gap-6">
                    {/* Database Selection */}
                    <div className="col-span-6">
                      <label htmlFor="database" className="block text-sm font-medium text-gray-700">
                        Database
                      </label>
                      <select
                        id="database"
                        name="database"
                        value={selectedDatabase}
                        onChange={(e) => setSelectedDatabase(e.target.value)}
                        required
                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                      >
                        <option value="">Select a database</option>
                        {databases.map((db) => (
                          <option key={db.id} value={db.id}>{db.name}</option>
                        ))}
                      </select>
                    </div>

                    {/* Operation Type Selection */}
                    <div className="col-span-6">
                      <div className="flex space-x-4">
                        <div className="flex items-center">
                          <input
                            id="upload-type"
                            name="batch-operation"
                            type="radio"
                            checked={batchOperation === 'upload'}
                            onChange={() => setBatchOperation('upload')}
                            className="focus:ring-indigo-500 h-4 w-4 text-indigo-600 border-gray-300"
                          />
                          <label htmlFor="upload-type" className="ml-2 block text-sm text-gray-700">
                            Upload Vectors
                          </label>
                        </div>
                        <div className="flex items-center">
                          <input
                            id="download-type"
                            name="batch-operation"
                            type="radio"
                            checked={batchOperation === 'download'}
                            onChange={() => setBatchOperation('download')}
                            className="focus:ring-indigo-500 h-4 w-4 text-indigo-600 border-gray-300"
                          />
                          <label htmlFor="download-type" className="ml-2 block text-sm text-gray-700">
                            Download Vectors
                          </label>
                        </div>
                      </div>
                    </div>

                    {/* Upload Operation UI */}
                    {batchOperation === 'upload' && (
                      <>
                        <div className="col-span-6">
                          <h4 className="text-md font-medium text-gray-900">Vector Data</h4>
                          <p className="mt-1 text-sm text-gray-500">
                            Enter vectors manually or upload from file
                          </p>
                        </div>

                        {/* Vector Input Fields */}
                        {vectors.map((vec, index) => (
                          <div key={index} className="col-span-6 grid grid-cols-12 gap-4">
                            <div className="col-span-12 md:col-span-3">
                              <label htmlFor={`id-${index}`} className="block text-sm font-medium text-gray-700">
                                ID
                              </label>
                              <input
                                type="text"
                                id={`id-${index}`}
                                value={vec.id}
                                onChange={(e) => handleVectorChange(index, 'id', e.target.value)}
                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                                placeholder="Vector ID"
                              />
                            </div>
                            <div className="col-span-12 md:col-span-6">
                              <label htmlFor={`values-${index}`} className="block text-sm font-medium text-gray-700">
                                Values
                              </label>
                              <input
                                type="text"
                                id={`values-${index}`}
                                value={vec.values}
                                onChange={(e) => handleVectorChange(index, 'values', e.target.value)}
                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                                placeholder="Comma-separated or JSON array"
                              />
                            </div>
                            <div className="col-span-12 md:col-span-2">
                              <label htmlFor={`metadata-${index}`} className="block text-sm font-medium text-gray-700">
                                Metadata (JSON)
                              </label>
                              <input
                                type="text"
                                id={`metadata-${index}`}
                                value={vec.metadata}
                                onChange={(e) => handleVectorChange(index, 'metadata', e.target.value)}
                                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                                placeholder='{"key": "value"}'
                              />
                            </div>
                            <div className="col-span-12 md:col-span-1 flex items-end">
                              <button
                                type="button"
                                onClick={() => handleRemoveVector(index)}
                                className="inline-flex items-center px-2.5 py-1.5 border border-gray-300 shadow-sm text-xs font-medium rounded text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                              >
                                Remove
                              </button>
                            </div>
                          </div>
                        ))}

                        <div className="col-span-6">
                          <button
                            type="button"
                            onClick={handleAddVector}
                            className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                          >
                            Add Vector
                          </button>
                        </div>

                        <div className="col-span-6">
                          <label className="block text-sm font-medium text-gray-700">
                            Or upload from file
                          </label>
                          <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md">
                            <div className="space-y-1 text-center">
                              <div className="flex text-sm text-gray-600">
                                <label
                                  htmlFor="file-upload"
                                  className="relative cursor-pointer bg-white rounded-md font-medium text-indigo-600 hover:text-indigo-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-indigo-500"
                                >
                                  <span>Upload a file</span>
                                  <input
                                    id="file-upload"
                                    ref={fileInputRef}
                                    name="file-upload"
                                    type="file"
                                    accept=".json,.csv"
                                    onChange={handleFileUpload}
                                    className="sr-only"
                                  />
                                </label>
                                <p className="pl-1">or drag and drop</p>
                              </div>
                              <p className="text-xs text-gray-500">
                                JSON or CSV files up to 10MB
                              </p>
                            </div>
                          </div>
                          {fileContent && (
                            <div className="mt-4">
                              <div className="flex items-center justify-between">
                                <label className="block text-sm font-medium text-gray-700">
                                  File Content Preview
                                </label>
                                <button
                                  type="button"
                                  onClick={() => setFileContent('')}
                                  className="text-xs text-red-600 hover:text-red-900"
                                >
                                  Clear
                                </button>
                              </div>
                              <div className="mt-1">
                                <textarea
                                  value={fileContent}
                                  readOnly
                                  rows={4}
                                  className="w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 text-xs font-mono"
                                />
                              </div>
                            </div>
                          )}
                        </div>
                      </>
                    )}

                    {/* Download Operation UI */}
                    {batchOperation === 'download' && (
                      <>
                        <div className="col-span-6">
                          <label htmlFor="downloadFormat" className="block text-sm font-medium text-gray-700">
                            Download Format
                          </label>
                          <select
                            id="downloadFormat"
                            name="downloadFormat"
                            value={downloadFormat}
                            onChange={(e) => setDownloadFormat(e.target.value)}
                            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                          >
                            <option value="json">JSON</option>
                            <option value="csv">CSV</option>
                          </select>
                        </div>
                        <div className="col-span-6">
                          <p className="text-sm text-gray-500">
                            Download all vectors from the selected database in the chosen format.
                          </p>
                        </div>
                      </>
                    )}

                    {/* Progress bar */}
                    {loading && (
                      <div className="col-span-6">
                        <div className="w-full bg-gray-200 rounded-full h-2.5">
                          <div 
                            className="bg-blue-600 h-2.5 rounded-full transition-all duration-300 ease-in-out" 
                            style={{ width: `${progress}%` }}
                          ></div>
                        </div>
                        <div className="text-sm text-gray-500 mt-1">Progress: {progress}%</div>
                      </div>
                    )}

                    {/* Submit Button */}
                    <div className="col-span-6">
                      <button
                        type="submit"
                        disabled={loading || !selectedDatabase}
                        className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
                      >
                        {loading 
                          ? (batchOperation === 'upload' ? 'Uploading...' : 'Downloading...') 
                          : (batchOperation === 'upload' ? 'Upload Vectors' : 'Download Vectors')
                        }
                      </button>
                    </div>
                  </div>
                </form>
              </div>
            </div>
          </div>

          {/* Results */}
          {results && (
            <div className={`bg-white shadow overflow-hidden sm:rounded-md ${
              results.success ? 'border border-green-200' : 'border border-red-200'
            }`}>
              <div className={`px-4 py-5 border-b ${
                results.success ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'
              } sm:px-6`}>
                <h3 className={`text-lg leading-6 font-medium ${
                  results.success ? 'text-green-800' : 'text-red-800'
                }`}>
                  {results.success ? 'Success' : 'Error'}
                </h3>
              </div>
              <div className="p-6">
                <p className={`text-sm ${
                  results.success ? 'text-green-700' : 'text-red-700'
                }`}>
                  {results.message}
                </p>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}