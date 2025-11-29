import React, { useState, useRef, useEffect } from 'react';
import Editor from '@monaco-editor/react';
import * as monaco from 'monaco-editor';
import { useTutorial } from '../contexts/TutorialContext';
import { getApiService } from '../services/api';

// Define custom themes for the editor
const defineCustomThemes = () => {
  // Custom dark theme
  monaco.editor.defineTheme('jade-dark', {
    base: 'vs-dark',
    inherit: true,
    rules: [
      { token: 'comment', foreground: '6A9955' },
      { token: 'keyword', foreground: '569CD6' },
      { token: 'string', foreground: 'CE9178' },
      { token: 'number', foreground: 'B5CEA8' },
      { token: 'function', foreground: 'DCDCAA' },
      { token: 'variable', foreground: '9CDCFE' },
      { token: 'type', foreground: '4EC9B0' },
    ],
    colors: {
      'editor.background': '#1E1E1E',
      'editor.lineHighlightBackground': '#2A2D2E',
      'editorLineNumber.foreground': '#858585',
      'editorCursor.foreground': '#AEAFAD',
    }
  });

  // Custom light theme
  monaco.editor.defineTheme('jade-light', {
    base: 'vs',
    inherit: true,
    rules: [
      { token: 'comment', foreground: '008000' },
      { token: 'keyword', foreground: '0000FF' },
      { token: 'string', foreground: 'A31515' },
      { token: 'number', foreground: '098658' },
      { token: 'function', foreground: '795E26' },
      { token: 'variable', foreground: '001080' },
      { token: 'type', foreground: '267F99' },
    ],
    colors: {
      'editor.background': '#FFFFFF',
      'editor.lineHighlightBackground': '#F5F5F5',
      'editorLineNumber.foreground': '#237893',
      'editorCursor.foreground': '#000000',
    }
  });
};

// Define custom language for JadeVectorDB API
const defineJadeVectorDBLanguage = () => {
  // Register a new language
  monaco.languages.register({ id: 'jadevectordb' });

  // Register a tokens provider for the language
  monaco.languages.setMonarchTokensProvider('jadevectordb', {
    tokenizer: {
      root: [
        [/\b(createDatabase|storeVector|search|storeVectorsBatch|getVector|updateVector|deleteVector|createIndex|listIndexes|deleteIndex)\b/, 'function'],
        [/\b(client|db|vector|queryVector|searchResults|vectors|index)\b/, 'variable'],
        [/\b(await|async|const|let|var|function|return|if|else|for|while|try|catch|finally)\b/, 'keyword'],
        [/"[^"]*"/, 'string'],
        [/'[^']*'/, 'string'],
        [/\b\d+\.?\d*\b/, 'number'],
        [/\/\*[^\*]*\*\//, 'comment'],
        [/\/\/.*/, 'comment'],
        [/[{}()\[\]]/, '@brackets'],
        [/[a-zA-Z_]\w*/, 'identifier']
      ]
    }
  });

  // Register a completion item provider for the language
  monaco.languages.registerCompletionItemProvider('jadevectordb', {
    provideCompletionItems: (model, position) => {
      const word = model.getWordUntilPosition(position);
      const range = {
        startLineNumber: position.lineNumber,
        endLineNumber: position.lineNumber,
        startColumn: word.startColumn,
        endColumn: word.endColumn
      };

      return {
        suggestions: [
          {
            label: 'client',
            kind: monaco.languages.CompletionItemKind.Variable,
            insertText: 'client',
            range: range
          },
          {
            label: 'createDatabase',
            kind: monaco.languages.CompletionItemKind.Function,
            insertText: 'createDatabase({\n\tname: "${1:databaseName}",\n\tvectorDimension: ${2:128},\n\tindexType: "${3:HNSW}"\n})',
            insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
            range: range
          },
          {
            label: 'storeVector',
            kind: monaco.languages.CompletionItemKind.Function,
            insertText: 'storeVector("${1:databaseId}", {\n\tid: "${2:vectorId}",\n\tvalues: [${3:0.1, 0.2, 0.3}],\n\tmetadata: {\n\t\t${4:key}: "${5:value}"\n\t}\n})',
            insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
            range: range
          },
          {
            label: 'search',
            kind: monaco.languages.CompletionItemKind.Function,
            insertText: 'search("${1:databaseId}", {\n\tvalues: [${2:0.1, 0.2, 0.3}]\n}, {\n\ttopK: ${3:10},\n\tthreshold: ${4:0.7}\n})',
            insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
            range: range
          }
        ]
      };
    }
  });
};

const AdvancedCodeEditor = () => {
  const [code, setCode] = useState('');
  const [language, setLanguage] = useState('javascript');
  const [theme, setTheme] = useState('jade-light');
  const [fontSize, setFontSize] = useState(14);
  const [wordWrap, setWordWrap] = useState('off');
  const [minimap, setMinimap] = useState(true);
  const [isRunning, setIsRunning] = useState(false);
  const [output, setOutput] = useState('');
  const [executionTime, setExecutionTime] = useState(0);
  const [lineNumbers, setLineNumbers] = useState('on');
  const [renderWhitespace, setRenderWhitespace] = useState('none');
  const editorRef = useRef(null);
  const { tutorialState, currentModule, currentStep } = useTutorial();

  // Initialize custom themes and language
  useEffect(() => {
    defineCustomThemes();
    defineJadeVectorDBLanguage();
  }, []);

  // Sample code examples based on the current module/step and selected language
  useEffect(() => {
    let sampleCode = '';
    
    // Generate appropriate code based on selected language
    if (language === 'python') {
      if (currentModule === 0) { // Getting Started
        if (currentStep === 0) {
          sampleCode = `# Create a new vector database
import jadevectordb

client = jadevectordb.Client(api_key="YOUR_API_KEY")

db = client.create_database(
    name="tutorial-database",
    vector_dimension=128,
    index_type="HNSW"
)

print(f"Database created: {db.id}")`;
        } else if (currentStep === 1) {
          sampleCode = `# Store a vector with metadata
vector = {
    "id": "vector-1",
    "values": [0.1, 0.2, 0.3] + [0.0] * 125,  # Pad to 128 dimensions
    "metadata": {
        "category": "example",
        "tags": ["tutorial", "vector"],
        "score": 0.95
    }
}

result = client.store_vector(db.id, vector)

print(f"Vector stored: {result.id}")`;
        } else {
          sampleCode = `# Perform similarity search
query_vector = [0.15, 0.25, 0.35] + [0.0] * 125  # Pad to 128 dimensions

search_results = client.search(
    database_id=db.id,
    query_vector=query_vector,
    top_k=5,
    threshold=0.7
)

print("Similar vectors:", search_results)`;
        }
      } else if (currentModule === 1) { // Vector Manipulation
        sampleCode = `# Batch store multiple vectors
vectors = [
    {
        "id": "vec-1",
        "values": [0.1, 0.2, 0.3] + [0.0] * 125,
        "metadata": {"tag": "example"}
    },
    {
        "id": "vec-2", 
        "values": [0.4, 0.5, 0.6] + [0.0] * 125,
        "metadata": {"tag": "example"}
    }
]

results = client.store_vectors_batch(db.id, vectors)

print(f"Batch stored: {len(results)} vectors")`;
      } else {
        sampleCode = `# Example API call
response = client.search(
    database_id=db.id,
    query_vector=[0.1, 0.2, 0.3] + [0.0] * 125,
    top_k=10,
    threshold=0.8,
    filters={"category": "example"}
)

print("Results:", response)`;
      }
    } else if (language === 'curl') {
      if (currentModule === 0) { // Getting Started
        if (currentStep === 0) {
          sampleCode = `# Create a new vector database
curl -X POST https://api.jadevectordb.com/v1/databases \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "name": "tutorial-database",
    "vectorDimension": 128,
    "indexType": "HNSW"
  }'`;
        } else if (currentStep === 1) {
          sampleCode = `# Store a vector with metadata
curl -X POST https://api.jadevectordb.com/v1/databases/db_tutorial_123/vectors \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "id": "vector-1",
    "values": [0.1, 0.2, 0.3, 0.0, 0.0],
    "metadata": {
      "category": "example",
      "tags": ["tutorial", "vector"],
      "score": 0.95
    }
  }'`;
        } else {
          sampleCode = `# Perform similarity search
curl -X POST https://api.jadevectordb.com/v1/databases/db_tutorial_123/search \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "queryVector": [0.15, 0.25, 0.35, 0.0, 0.0],
    "topK": 5,
    "threshold": 0.7
  }'`;
        }
      } else {
        sampleCode = `# Example API call
curl -X POST https://api.jadevectordb.com/v1/databases/db_tutorial_123/search \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "queryVector": [0.1, 0.2, 0.3, 0.0, 0.0],
    "topK": 10,
    "threshold": 0.8,
    "filters": {
      "category": "example"
    }
  }'`;
      }
    } else if (language === 'jadevectordb') {
      sampleCode = `// JadeVectorDB API example
// Create database
const db = await apiService.createDatabase({
  name: "tutorial-database",
  vectorDimension: 128,
  indexType: "HNSW"
});

console.log("Database created:", db);

// Store vector
const vector = {
  id: "vector-1",
  values: [0.1, 0.2, 0.3] /* ... more values ... */,
  metadata: {
    category: "example",
    tags: ["tutorial", "vector"],
    score: 0.95
  }
};

const result = await apiService.storeVector(db.databaseId, vector);
console.log("Vector stored:", result);

// Search
const queryVector = [0.15, 0.25, 0.35 /* ... */];
const searchResults = await apiService.similaritySearch(db.databaseId, { values: queryVector }, {
  topK: 5,
  threshold: 0.7
});
console.log("Search results:", searchResults);`;
    } else {
      // Default to JavaScript using the real API service
      if (currentModule === 0) { // Getting Started
        if (currentStep === 0) {
          sampleCode = `// Create a new vector database
const db = await apiService.createDatabase({
  name: "tutorial-database",
  vectorDimension: 128,
  indexType: "HNSW"
});

console.log("Database created:", db);`;
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

const result = await apiService.storeVector("tutorial-database", vector);

console.log("Vector stored:", result);`;
        } else {
          sampleCode = `// Perform similarity search
const queryVector = [0.15, 0.25, 0.35, /* ... */];

const searchResults = await apiService.similaritySearch("tutorial-database", { values: queryVector }, {
  topK: 5,
  threshold: 0.7
});

console.log("Search results:", searchResults);`;
        }
      } else if (currentModule === 1) { // Vector Manipulation
        sampleCode = `// Batch store multiple vectors
const vectors = [
  {
    id: "vec-1",
    values: [0.1, 0.2, 0.3, /* ... more values ... */],
    metadata: { tag: "example" }
  },
  {
    id: "vec-2", 
    values: [0.4, 0.5, 0.6, /* ... more values ... */],
    metadata: { tag: "example" }
  }
];

const results = await apiService.batchStoreVectors("tutorial-database", vectors);

console.log("Batch stored:", results);`;
      } else {
        sampleCode = `// Example API call
const response = await apiService.similaritySearch("tutorial-database", 
  { values: [0.1, 0.2, 0.3, /* ... more values ... */] }, 
  {
    topK: 10,
    threshold: 0.8,
    filters: {
      category: "example"
    }
  }
);
console.log("Results:", response);`;
      }
    }
    
    setCode(sampleCode);
  }, [currentModule, currentStep, language]);

  const handleEditorDidMount = (editor, monacoInstance) => {
    editorRef.current = editor;
    
    // Add custom keybindings
    editor.addAction({
      id: 'run-code',
      label: 'Run Code',
      keybindings: [
        monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter,
      ],
      run: () => handleRunCode()
    });
    
    editor.addAction({
      id: 'format-code',
      label: 'Format Code',
      keybindings: [
        monaco.KeyMod.CtrlCmd | monaco.KeyCode.KEY_F,
      ],
      run: () => editor.getAction('editor.action.formatDocument').run()
    });
    
    editor.addAction({
      id: 'toggle-comment',
      label: 'Toggle Comment',
      keybindings: [
        monaco.KeyMod.CtrlCmd | monaco.KeyCode.US_SLASH,
      ],
      run: () => editor.getAction('editor.action.commentLine').run()
    });
  };

  // Get the API service instance
  const apiService = getApiService();

  const handleRunCode = async () => {
    setIsRunning(true);
    setOutput('');
    
    try {
      const startTime = Date.now();
      
      // Parse and execute the code based on language
      let resultOutput = '';
      
      if (language === 'javascript' || language === 'jadevectordb') {
        // For JavaScript, we'll execute based on function names in the code
        if (code.includes('createDatabase')) {
          const match = code.match(/createDatabase\(\s*{[^}]*name\s*:\s*["']([^"']+)["'][^}]*}/);
          if (match) {
            const dbName = match[1];
            try {
              const db = await apiService.createDatabase({ 
                name: dbName, 
                vectorDimension: 128, 
                indexType: "HNSW" 
              });
              resultOutput = `Database created successfully\n${JSON.stringify(db, null, 2)}`;
            } catch (err) {
              resultOutput = `Error creating database: ${err.message}`;
            }
          }
        } else if (code.includes('storeVector')) {
          // Extract database ID and vector data
          const dbMatch = code.match(/storeVector\(\s*["']([^"']+)["']/);
          const idMatch = code.match(/id\s*:\s*["']([^"']+)["']/);
          const valuesMatch = code.match(/values\s*:\s*\[([^\]]+)\]/);
          
          if (dbMatch && idMatch && valuesMatch) {
            const dbId = dbMatch[1];
            const vectorId = idMatch[1];
            const valuesStr = valuesMatch[1];
            const values = valuesStr.split(',').map(v => parseFloat(v.trim()));
            
            try {
              const vector = {
                id: vectorId,
                values: values,
                metadata: { category: "example", timestamp: new Date().toISOString() }
              };
              const result = await apiService.storeVector(dbId, vector);
              resultOutput = `Vector stored successfully\n${JSON.stringify(result, null, 2)}`;
            } catch (err) {
              resultOutput = `Error storing vector: ${err.message}`;
            }
          }
        } else if (code.includes('apiService.similaritySearch') || code.includes('search(')) {
          // Extract database ID and query vector
          const dbMatch = code.match(/similaritySearch\(\s*["']([^"']+)["']/);
          const valuesMatch = /values\s*:\s*\[([^\]]+)\]/.exec(code);
          
          if (dbMatch && valuesMatch) {
            const dbId = dbMatch[1];
            const valuesStr = valuesMatch[1];
            const values = valuesStr.split(',').map(v => parseFloat(v.trim()));
            
            try {
              const queryVector = { values };
              const searchParams = {};
              // Extract topK if present
              const topKMatch = code.match(/topK\s*:\s*(\d+)/);
              if (topKMatch) searchParams.topK = parseInt(topKMatch[1]);
              
              // Extract threshold if present
              const thresholdMatch = code.match(/threshold\s*:\s*([\d.]+)/);
              if (thresholdMatch) searchParams.threshold = parseFloat(thresholdMatch[1]);
              
              const results = await apiService.similaritySearch(dbId, queryVector, searchParams);
              resultOutput = `Search completed successfully\n${JSON.stringify(results, null, 2)}`;
            } catch (err) {
              resultOutput = `Error performing search: ${err.message}`;
            }
          }
        } else {
          resultOutput = `Code executed. No recognizable JadeVectorDB operation found in code.`;
        }
      } else if (language === 'python') {
        resultOutput = `Python execution would require a backend service to run the code.\nFor now, the tutorial focuses on JavaScript execution.\n\nExecuted code:\n${code}`;
      } else if (language === 'curl') {
        resultOutput = `cURL execution would require a backend service to run the command.\nFor now, the tutorial focuses on JavaScript execution.\n\nExecuted command:\n${code}`;
      } else {
        resultOutput = `Language ${language} execution not yet implemented. Only JavaScript is currently supported for direct execution.`;
      }
      
      const endTime = Date.now();
      setExecutionTime(endTime - startTime);
      setOutput(resultOutput);
    } catch (error) {
      const endTime = Date.now();
      setExecutionTime(endTime - startTime);
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

  const handleToggleComment = () => {
    if (editorRef.current) {
      editorRef.current.getAction('editor.action.commentLine').run();
    }
  };

  const handleIndentMore = () => {
    if (editorRef.current) {
      editorRef.current.getAction('editor.action.indentLines').run();
    }
  };

  const handleIndentLess = () => {
    if (editorRef.current) {
      editorRef.current.getAction('editor.action.outdentLines').run();
    }
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
              <option value="jadevectordb">JadeVectorDB</option>
            </select>
          </div>
          
          <div className="flex items-center space-x-2">
            <label className="text-sm text-gray-700">Theme:</label>
            <select 
              value={theme} 
              onChange={(e) => setTheme(e.target.value)}
              className="p-1 border border-gray-300 rounded-md text-sm"
            >
              <option value="jade-light">Jade Light</option>
              <option value="jade-dark">Jade Dark</option>
              <option value="vs-light">VS Light</option>
              <option value="vs-dark">VS Dark</option>
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
              <option value="20">20px</option>
            </select>
          </div>
        </div>
      </div>
      
      <div className="border border-gray-200 rounded-lg overflow-hidden">
        <Editor
          key={language} // Add key to force re-render when language changes
          height="400px"
          language={language}
          value={code}
          onChange={(value) => setCode(value || '')}
          theme={theme}
          onMount={handleEditorDidMount}
          options={{
            minimap: { enabled: minimap },
            fontSize: fontSize,
            scrollBeyondLastLine: false,
            automaticLayout: true,
            tabSize: 2,
            suggestOnTriggerCharacters: true,
            wordBasedSuggestions: true,
            quickSuggestions: true,
            wordWrap: wordWrap,
            lineNumbers: lineNumbers,
            renderWhitespace: renderWhitespace,
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
        
        <div className="relative group">
          <button className="btn-secondary flex items-center px-4 py-2">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            Settings
          </button>
          
          <div className="absolute right-0 mt-1 w-64 bg-white rounded-md shadow-lg py-1 hidden group-hover:block z-10 border border-gray-200">
            <div className="px-4 py-2 border-b border-gray-100">
              <h3 className="text-sm font-medium text-gray-800">Editor Settings</h3>
            </div>
            
            <div className="px-4 py-2">
              <div className="flex items-center justify-between py-1">
                <label className="text-sm text-gray-700">Word Wrap</label>
                <select 
                  value={wordWrap} 
                  onChange={(e) => setWordWrap(e.target.value)}
                  className="p-1 border border-gray-300 rounded-md text-sm"
                >
                  <option value="off">Off</option>
                  <option value="on">On</option>
                </select>
              </div>
              
              <div className="flex items-center justify-between py-1">
                <label className="text-sm text-gray-700">Minimap</label>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input 
                    type="checkbox" 
                    checked={minimap} 
                    onChange={(e) => setMinimap(e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
                </label>
              </div>
              
              <div className="flex items-center justify-between py-1">
                <label className="text-sm text-gray-700">Line Numbers</label>
                <select 
                  value={lineNumbers} 
                  onChange={(e) => setLineNumbers(e.target.value)}
                  className="p-1 border border-gray-300 rounded-md text-sm"
                >
                  <option value="on">On</option>
                  <option value="off">Off</option>
                  <option value="relative">Relative</option>
                </select>
              </div>
              
              <div className="flex items-center justify-between py-1">
                <label className="text-sm text-gray-700">Whitespace</label>
                <select 
                  value={renderWhitespace} 
                  onChange={(e) => setRenderWhitespace(e.target.value)}
                  className="p-1 border border-gray-300 rounded-md text-sm"
                >
                  <option value="none">None</option>
                  <option value="boundary">Boundary</option>
                  <option value="all">All</option>
                </select>
              </div>
            </div>
          </div>
        </div>
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
        <h3 className="font-medium text-blue-800 mb-2">Keyboard Shortcuts</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2 text-sm">
          <div className="flex items-center">
            <kbd className="px-2 py-1 text-xs font-semibold text-gray-800 bg-gray-100 border border-gray-200 rounded-lg">Ctrl+Enter</kbd>
            <span className="ml-2 text-gray-700">Run Code</span>
          </div>
          <div className="flex items-center">
            <kbd className="px-2 py-1 text-xs font-semibold text-gray-800 bg-gray-100 border border-gray-200 rounded-lg">Ctrl+F</kbd>
            <span className="ml-2 text-gray-700">Format Code</span>
          </div>
          <div className="flex items-center">
            <kbd className="px-2 py-1 text-xs font-semibold text-gray-800 bg-gray-100 border border-gray-200 rounded-lg">Ctrl+/</kbd>
            <span className="ml-2 text-gray-700">Toggle Comment</span>
          </div>
        </div>
      </div>
      
      <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-100">
        <h3 className="font-medium text-blue-800 mb-2">API Documentation</h3>
        <p className="text-sm text-blue-700">
          Need help with the API? Check out the{' '}
          <a href="#" className="text-blue-600 hover:underline">API Reference</a>{' '}
          for detailed information about each method and its parameters.
        </p>
        <div className="mt-2">
          <p className="text-xs text-blue-600 mb-2">
            <strong>Note:</strong> JadeVectorDB CLI is currently available for Python and shell scripts only. 
            Other language examples are provided for reference.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
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

export default AdvancedCodeEditor;