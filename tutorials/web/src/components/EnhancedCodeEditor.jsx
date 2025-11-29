import React, { useState, useRef, useEffect } from 'react';
import Editor from '@monaco-editor/react';
import { useTutorial } from '../contexts/TutorialContext';

const EnhancedCodeEditor = () => {
  const [code, setCode] = useState('');
  const [language, setLanguage] = useState('javascript');
  const [theme, setTheme] = useState('vs-light');
  const [fontSize, setFontSize] = useState(14);
  const [isRunning, setIsRunning] = useState(false);
  const [output, setOutput] = useState('');
  const [executionTime, setExecutionTime] = useState(0);
  const editorRef = useRef(null);
  const { tutorialState, currentModule, currentStep } = useTutorial();

  // Sample code examples based on the current module/step
  useEffect(() => {
    let sampleCode = '';
    
    if (currentModule === 0) { // Getting Started
      if (currentStep === 0) {
        sampleCode = `// Create a new vector database
const db = await client.createDatabase({
  name: "tutorial-database",
  vectorDimension: 128,
  indexType: "HNSW"
});

console.log("Database created:", db.id);`;
      } else if (currentStep === 1) {
        sampleCode = `// Store a vector with metadata
const vector = {
  id: "vector-1",
  values: [0.1, 0.2, 0.3, /* ... more values ... */],
  metadata: {
    category: "example",
    tags: ["tutorial", "vector"],
    score: 0.95
  }
};

const result = await client.storeVector(db.id, vector);

console.log("Vector stored:", result.id);`;
      } else {
        sampleCode = `// Perform similarity search
const queryVector = {
  values: [0.15, 0.25, 0.35, /* ... */]
};

const searchResults = await client.search(db.id, queryVector, {
  topK: 5,
  threshold: 0.7
});

console.log("Similar vectors:", searchResults);`;
      }
    } else if (currentModule === 1) { // Vector Manipulation
      sampleCode = `// Batch store multiple vectors
const vectors = [
  {
    id: "vec-1",
    values: [0.1, 0.2, 0.3],
    metadata: { tag: "example" }
  },
  {
    id: "vec-2", 
    values: [0.4, 0.5, 0.6],
    metadata: { tag: "example" }
  }
];

const results = await client.storeVectorsBatch(db.id, vectors);

console.log("Batch stored:", results.length, "vectors");`;
    } else {
      sampleCode = `// Example API call
const response = await client.search(db.id, queryVector, {
  topK: 10,
  threshold: 0.8,
  filters: {
    category: "example"
  }
});

console.log("Results:", response);`;
    }
    
    setCode(sampleCode);
  }, [currentModule, currentStep]);

  const handleEditorDidMount = (editor, monaco) => {
    editorRef.current = editor;
  };

  const handleRunCode = async () => {
    setIsRunning(true);
    setOutput('');
    
    try {
      // Simulate code execution with a delay
      const startTime = Date.now();
      
      // In a real implementation, this would send the code to the backend simulation
      // For now, we'll simulate different outputs based on the code
      await new Promise(resolve => setTimeout(resolve, 800));
      
      const endTime = Date.now();
      setExecutionTime(endTime - startTime);
      
      // Simulate different outputs based on code content
      if (code.includes('createDatabase')) {
        setOutput(`Database created successfully
ID: db_tutorial_123456
Name: tutorial-database
Vector Dimension: 128
Index Type: HNSW
Status: active`);
      } else if (code.includes('storeVector')) {
        setOutput(`Vector stored successfully
ID: vector-1
Similarity: 1.00
Metadata: {
  category: "example",
  tags: ["tutorial", "vector"],
  score: 0.95
}`);
      } else if (code.includes('search')) {
        setOutput(`Search completed successfully
Results found: 3
Top result: vec-result-1 (similarity: 0.85)
Second result: vec-result-2 (similarity: 0.72)
Third result: vec-result-3 (similarity: 0.68)`);
      } else {
        setOutput(`Code executed successfully
No specific output generated`);
      }
    } catch (error) {
      setOutput(`Error: ${error.message}`);
    } finally {
      setIsRunning(false);
    }
  };

  const handleResetCode = () => {
    // Reset to the original sample code
    setOutput('');
    setExecutionTime(0);
    
    // Trigger useEffect to reset code
    const event = new Event('resetCode');
    window.dispatchEvent(event);
  };

  const handleFormatCode = () => {
    if (editorRef.current) {
      editorRef.current.getAction('editor.action.formatDocument').run();
    }
  };

  const handleCopyCode = () => {
    navigator.clipboard.writeText(code);
    // Show a brief confirmation
    const button = document.getElementById('copy-button');
    if (button) {
      const originalText = button.textContent;
      button.textContent = 'Copied!';
      setTimeout(() => {
        button.textContent = originalText;
      }, 2000);
    }
  };

  const handleExportCode = () => {
    // Create a blob with the code content
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    
    // Create a temporary link and trigger download
    const a = document.createElement('a');
    a.href = url;
    a.download = `jadevectordb-tutorial-${currentModule}-${currentStep}.${language === 'javascript' ? 'js' : language === 'python' ? 'py' : 'txt'}`;
    document.body.appendChild(a);
    a.click();
    
    // Clean up
    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 100);
  };

  return (
    <div className="module-card">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-4">
        <h2 className="text-xl font-semibold text-gray-800 mb-2 md:mb-0">Code Editor</h2>
        
        <div className="flex flex-wrap gap-2">
          <div className="flex items-center space-x-2">
            <label className="text-sm text-gray-700">Language:</label>
            <select 
              value={language} 
              onChange={(e) => setLanguage(e.target.value)}
              className="p-1 border border-gray-300 rounded-md text-sm"
            >
              <option value="javascript">JavaScript</option>
              <option value="python">Python</option>
              <option value="go">Go</option>
              <option value="java">Java</option>
              <option value="curl">cURL</option>
            </select>
          </div>
          
          <div className="flex items-center space-x-2">
            <label className="text-sm text-gray-700">Theme:</label>
            <select 
              value={theme} 
              onChange={(e) => setTheme(e.target.value)}
              className="p-1 border border-gray-300 rounded-md text-sm"
            >
              <option value="vs-light">Light</option>
              <option value="vs-dark">Dark</option>
              <option value="hc-black">High Contrast</option>
            </select>
          </div>
          
          <div className="flex items-center space-x-2">
            <label className="text-sm text-gray-700">Font:</label>
            <select 
              value={fontSize} 
              onChange={(e) => setFontSize(Number(e.target.value))}
              className="p-1 border border-gray-300 rounded-md text-sm"
            >
              <option value="12">12px</option>
              <option value="14">14px</option>
              <option value="16">16px</option>
              <option value="18">18px</option>
            </select>
          </div>
        </div>
      </div>
      
      <div className="border border-gray-200 rounded-lg overflow-hidden">
        <Editor
          height="400px"
          language={language}
          value={code}
          onChange={(value) => setCode(value || '')}
          theme={theme}
          onMount={handleEditorDidMount}
          options={{
            minimap: { enabled: false },
            fontSize: fontSize,
            scrollBeyondLastLine: false,
            automaticLayout: true,
            tabSize: 2,
            suggestOnTriggerCharacters: true,
            wordBasedSuggestions: true,
            quickSuggestions: true,
            suggest: {
              snippetsPreventQuickSuggestions: false
            }
          }}
        />
      </div>
      
      <div className="flex flex-wrap gap-2 mt-4">
        <button 
          onClick={handleRunCode}
          disabled={isRunning}
          className="btn-primary flex items-center px-4 py-2"
        >
          {isRunning ? (
            <>
              <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Running...
            </>
          ) : (
            <>
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Run Code
            </>
          )}
        </button>
        
        <button 
          onClick={handleResetCode}
          className="btn-secondary flex items-center px-4 py-2"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Reset
        </button>
        
        <button 
          onClick={handleFormatCode}
          className="btn-secondary flex items-center px-4 py-2"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
          </svg>
          Format
        </button>
        
        <button 
          id="copy-button"
          onClick={handleCopyCode}
          className="btn-secondary flex items-center px-4 py-2"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
          Copy
        </button>
        
        <button 
          onClick={handleExportCode}
          className="btn-secondary flex items-center px-4 py-2"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          Export
        </button>
      </div>
      
      {output && (
        <div className="mt-4 border border-gray-200 rounded-lg bg-gray-50">
          <div className="flex justify-between items-center px-4 py-2 bg-gray-100 border-b border-gray-200">
            <h3 className="font-medium text-gray-800">Output</h3>
            <div className="text-sm text-gray-600">
              Execution time: {executionTime}ms
            </div>
          </div>
          <div className="p-4 font-mono text-sm whitespace-pre-wrap bg-black text-green-400 max-h-40 overflow-y-auto">
            {output}
          </div>
        </div>
      )}
      
      <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-100">
        <h3 className="font-medium text-blue-800 mb-2">API Documentation</h3>
        <p className="text-sm text-blue-700">
          Need help with the API? Check out the{' '}
          <a href="#" className="text-blue-600 hover:underline">API Reference</a>{' '}
          for detailed information about each method and its parameters.
        </p>
        <div className="mt-2 flex flex-wrap gap-2">
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
            createDatabase()
          </span>
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
            storeVector()
          </span>
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
            search()
          </span>
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
            storeVectorsBatch()
          </span>
        </div>
      </div>
    </div>
  );
};

export default EnhancedCodeEditor;