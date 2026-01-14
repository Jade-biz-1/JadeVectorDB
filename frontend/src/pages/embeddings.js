import { useState } from 'react';
import Layout from '../components/Layout';
import { embeddingApi } from '../lib/api';

export default function EmbeddingGeneration() {
  const [inputText, setInputText] = useState('');
  const [inputImage, setInputImage] = useState(null);
  const [selectedModel, setSelectedModel] = useState('all-MiniLM-L6-v2');
  const [generatedEmbedding, setGeneratedEmbedding] = useState(null);
  const [loading, setLoading] = useState(false);
  const [embeddingType, setEmbeddingType] = useState('text'); // 'text' or 'image'
  
  // Available embedding models
  const embeddingModels = [
    { id: 'all-MiniLM-L6-v2', name: 'all-MiniLM-L6-v2', description: 'Sentence transformer model', inputType: 'text', outputDimension: 384 },
    { id: 'all-mpnet-base-v2', name: 'all-mpnet-base-v2', description: 'Sentence transformer model', inputType: 'text', outputDimension: 768 },
    { id: 'clip-ViT-B-32', name: 'clip-ViT-B-32', description: 'CLIP model for text and images', inputType: 'image', outputDimension: 512 },
    { id: 'openai-ada-002', name: 'openai-ada-002', description: 'OpenAI text embedding model', inputType: 'text', outputDimension: 1536 },
  ];

  const handleGenerateEmbedding = async (e) => {
    e.preventDefault();
    setLoading(true);
    setGeneratedEmbedding(null);
    
    try {
      let requestData;
      
      if (embeddingType === 'text') {
        requestData = {
          input: inputText,
          model: selectedModel,
          inputType: 'text'
        };
      } else { // image
        // For image embeddings, we'll simulate reading a file
        // In a real implementation, we'd process the uploaded image file
        requestData = {
          input: inputImage ? inputImage.name : 'sample_image.jpg',
          model: selectedModel,
          inputType: 'image'
        };
      }
      
      const result = await embeddingApi.generateEmbedding(requestData);
      setGeneratedEmbedding(result.embedding);
    } catch (error) {
      console.error('Error generating embedding:', error);
      alert(`Error generating embedding: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setInputImage(file);
    }
  };

  // Filter models based on input type
  const availableModels = embeddingModels.filter(model => 
    embeddingType === 'text' ? model.inputType === 'text' : model.inputType === 'image'
  );

  return (
    <Layout title="Embedding Generation - JadeVectorDB">
      <main>
        <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          {/* Embedding Generation Form */}
          <div className="bg-white shadow px-4 py-5 sm:rounded-lg sm:p-6 mb-8">
            <div className="md:grid md:grid-cols-3 md:gap-6">
              <div className="md:col-span-1">
                <h3 className="text-lg font-medium leading-6 text-gray-900">Generate Embedding</h3>
                <p className="mt-1 text-sm text-gray-500">
                  Generate vector embeddings from text or images.
                </p>
              </div>
              <div className="mt-5 md:mt-0 md:col-span-2">
                <form onSubmit={handleGenerateEmbedding}>
                  <div className="grid grid-cols-6 gap-6">
                    {/* Embedding Type Selection */}
                    <div className="col-span-6">
                      <div className="flex space-x-4">
                        <div className="flex items-center">
                          <input
                            id="text-type"
                            name="embedding-type"
                            type="radio"
                            checked={embeddingType === 'text'}
                            onChange={() => setEmbeddingType('text')}
                            className="focus:ring-indigo-500 h-4 w-4 text-indigo-600 border-gray-300"
                          />
                          <label htmlFor="text-type" className="ml-2 block text-sm text-gray-700">
                            Text
                          </label>
                        </div>
                        <div className="flex items-center">
                          <input
                            id="image-type"
                            name="embedding-type"
                            type="radio"
                            checked={embeddingType === 'image'}
                            onChange={() => setEmbeddingType('image')}
                            className="focus:ring-indigo-500 h-4 w-4 text-indigo-600 border-gray-300"
                          />
                          <label htmlFor="image-type" className="ml-2 block text-sm text-gray-700">
                            Image
                          </label>
                        </div>
                      </div>
                    </div>

                    {/* Text Input */}
                    {embeddingType === 'text' && (
                      <div className="col-span-6">
                        <label htmlFor="inputText" className="block text-sm font-medium text-gray-700">
                          Input Text
                        </label>
                        <textarea
                          id="inputText"
                          name="inputText"
                          rows={4}
                          value={inputText}
                          onChange={(e) => setInputText(e.target.value)}
                          required
                          placeholder="Enter text to generate embedding for..."
                          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                        />
                        <p className="mt-1 text-sm text-gray-500">
                          Enter text to convert to vector embedding
                        </p>
                      </div>
                    )}

                    {/* Image Input */}
                    {embeddingType === 'image' && (
                      <div className="col-span-6">
                        <label className="block text-sm font-medium text-gray-700">
                          Upload Image
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
                                  name="file-upload"
                                  type="file"
                                  accept="image/*"
                                  onChange={handleFileChange}
                                  className="sr-only"
                                />
                              </label>
                              <p className="pl-1">or drag and drop</p>
                            </div>
                            <p className="text-xs text-gray-500">
                              PNG, JPG, GIF up to 10MB
                            </p>
                          </div>
                        </div>
                        {inputImage && (
                          <p className="mt-1 text-sm text-gray-500">
                            Selected: {inputImage.name}
                          </p>
                        )}
                      </div>
                    )}

                    {/* Model Selection */}
                    <div className="col-span-6">
                      <label htmlFor="model" className="block text-sm font-medium text-gray-700">
                        Embedding Model
                      </label>
                      <select
                        id="model"
                        name="model"
                        value={selectedModel}
                        onChange={(e) => setSelectedModel(e.target.value)}
                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
                      >
                        {availableModels.map((model) => (
                          <option key={model.id} value={model.id}>
                            {model.name} ({model.outputDimension}D) - {model.description}
                          </option>
                        ))}
                      </select>
                      <p className="mt-1 text-sm text-gray-500">
                        Select an embedding model to use
                      </p>
                    </div>
                  </div>

                  <div className="mt-6">
                    <button
                      type="submit"
                      disabled={loading || (embeddingType === 'text' && !inputText) || (embeddingType === 'image' && !inputImage)}
                      className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
                    >
                      {loading ? 'Generating...' : 'Generate Embedding'}
                    </button>
                  </div>
                </form>
              </div>
            </div>
          </div>

          {/* Embedding Results */}
          {generatedEmbedding && (
            <div className="bg-white shadow overflow-hidden sm:rounded-md">
              <div className="px-4 py-5 border-b border-gray-200 sm:px-6">
                <h3 className="text-lg leading-6 font-medium text-gray-900">Generated Embedding</h3>
                <p className="mt-1 text-sm text-gray-500">
                  Embedding generated with model: {selectedModel}
                </p>
              </div>
              <div className="p-6">
                <div className="mb-4">
                  <span className="text-sm font-medium text-gray-700">Embedding Dimensions:</span>
                  <span className="ml-2 text-sm text-gray-900">{generatedEmbedding.length}</span>
                </div>
                <div>
                  <span className="text-sm font-medium text-gray-700">Embedding Values (first 10):</span>
                  <div className="mt-2 p-4 bg-gray-100 rounded-md overflow-x-auto">
                    <pre className="text-xs text-gray-800 font-mono">
                      [{generatedEmbedding.slice(0, 10).map(v => v.toFixed(4)).join(', ')}{generatedEmbedding.length > 10 ? ', ...' : ''}]
                    </pre>
                  </div>
                </div>
                <div className="mt-4 flex space-x-3">
                  <button
                    onClick={() => {
                      navigator.clipboard.writeText(JSON.stringify(generatedEmbedding));
                      alert('Embedding copied to clipboard!');
                    }}
                    className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                  >
                    Copy to Clipboard
                  </button>
                  <button
                    onClick={() => {
                      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(generatedEmbedding));
                      const downloadAnchorNode = document.createElement('a');
                      downloadAnchorNode.setAttribute("href", dataStr);
                      downloadAnchorNode.setAttribute("download", `embedding_${selectedModel}.json`);
                      document.body.appendChild(downloadAnchorNode);
                      downloadAnchorNode.click();
                      downloadAnchorNode.remove();
                    }}
                    className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                  >
                    Download JSON
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </Layout>
  );
}