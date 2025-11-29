import React, { useState } from 'react';

/**
 * CodeChallengeQuestion - Code challenge question component
 *
 * Allows users to write code and validates against test cases.
 */
const CodeChallengeQuestion = ({
  question,
  answer,
  onChange,
  readOnly = false,
  showExplanation = false
}) => {
  const [code, setCode] = useState(answer || question.codeTemplate || '');
  const [testResults, setTestResults] = useState(null);
  const [isValidating, setIsValidating] = useState(false);

  /**
   * Handle code change
   */
  const handleCodeChange = (e) => {
    const newCode = e.target.value;
    setCode(newCode);

    // Try to parse as JSON object for config-type questions
    try {
      const parsed = parseCodeAnswer(newCode);
      onChange(parsed);
    } catch (err) {
      // If parsing fails, save as string
      onChange(newCode);
    }
  };

  /**
   * Parse code answer - try to extract the configuration object
   */
  const parseCodeAnswer = (codeString) => {
    // Try to extract object literal or configuration
    const lines = codeString.split('\n');

    // Look for object patterns
    const objMatch = codeString.match(/\{[\s\S]*\}/);
    if (objMatch) {
      // Clean up the object string
      let objStr = objMatch[0];

      // Try to convert to valid JSON
      // Replace property names without quotes
      objStr = objStr.replace(/(\w+):/g, '"$1":');
      // Replace single quotes with double quotes
      objStr = objStr.replace(/'/g, '"');

      try {
        return JSON.parse(objStr);
      } catch (e) {
        // If JSON parsing fails, try to evaluate
        return evaluateCode(codeString);
      }
    }

    return codeString;
  };

  /**
   * Evaluate code (simplified - in production use a sandbox)
   */
  const evaluateCode = (codeString) => {
    try {
      // Extract configuration object if present
      const configMatch = codeString.match(/\{[\s\S]*\}/);
      if (configMatch) {
        const objStr = configMatch[0]
          .replace(/(\w+):/g, '"$1":')
          .replace(/'/g, '"');
        return JSON.parse(objStr);
      }
      return codeString;
    } catch (error) {
      return codeString;
    }
  };

  /**
   * Run validation/tests
   */
  const handleValidate = () => {
    setIsValidating(true);

    try {
      const parsedAnswer = parseCodeAnswer(code);

      // If question has test cases, run them
      if (question.testCases && question.testCases.length > 0) {
        const results = question.testCases.map(testCase => {
          try {
            // Create context for test evaluation
            const context = {
              config: parsedAnswer,
              params: parsedAnswer,
              vector: parsedAnswer,
              update: parsedAnswer
            };

            // Evaluate test expression
            const func = new Function(...Object.keys(context), `return ${testCase.validate}`);
            const passed = func(...Object.values(context));

            return {
              description: testCase.description,
              passed: passed === true,
              error: null
            };
          } catch (error) {
            return {
              description: testCase.description,
              passed: false,
              error: error.message
            };
          }
        });

        setTestResults(results);
      } else {
        // Simple validation - just check if code is not empty
        setTestResults([{
          description: 'Code provided',
          passed: code.trim().length > 0,
          error: null
        }]);
      }
    } catch (error) {
      setTestResults([{
        description: 'Code validation',
        passed: false,
        error: error.message
      }]);
    }

    setIsValidating(false);
  };

  /**
   * Get test results summary
   */
  const getTestSummary = () => {
    if (!testResults) return null;

    const passed = testResults.filter(r => r.passed).length;
    const total = testResults.length;
    const allPassed = passed === total;

    return {
      passed,
      total,
      allPassed,
      percentage: Math.round((passed / total) * 100)
    };
  };

  const testSummary = getTestSummary();

  return (
    <div className="space-y-4">
      {/* Instructions */}
      {question.codeTemplate && (
        <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <p className="text-sm text-blue-800">
            💻 Complete the code below. A template has been provided to get you started.
          </p>
        </div>
      )}

      {/* Code editor */}
      <div className="relative">
        <textarea
          value={code}
          onChange={handleCodeChange}
          readOnly={readOnly}
          placeholder="Write your code here..."
          className="w-full h-64 px-4 py-3 font-mono text-sm border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none resize-y"
          spellCheck="false"
        />

        {/* Line numbers overlay (simplified) */}
        <div className="absolute left-2 top-3 text-gray-400 font-mono text-sm pointer-events-none select-none">
          {code.split('\n').map((_, i) => (
            <div key={i} className="leading-6">{i + 1}</div>
          ))}
        </div>
      </div>

      {/* Validate button */}
      {!readOnly && !showExplanation && (
        <button
          onClick={handleValidate}
          disabled={isValidating || !code.trim()}
          className={`px-4 py-2 rounded-lg font-medium ${
            isValidating || !code.trim()
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-green-600 text-white hover:bg-green-700'
          }`}
        >
          {isValidating ? 'Validating...' : '✓ Validate Code'}
        </button>
      )}

      {/* Test results */}
      {testResults && (
        <div className="space-y-3">
          {/* Summary */}
          {testSummary && (
            <div className={`p-4 rounded-lg border-2 ${
              testSummary.allPassed
                ? 'bg-green-50 border-green-300'
                : 'bg-yellow-50 border-yellow-300'
            }`}>
              <div className="flex items-center justify-between">
                <div>
                  <h4 className={`font-semibold ${
                    testSummary.allPassed ? 'text-green-900' : 'text-yellow-900'
                  }`}>
                    {testSummary.allPassed ? '✓ All Tests Passed!' : '⚠ Some Tests Failed'}
                  </h4>
                  <p className={`text-sm ${
                    testSummary.allPassed ? 'text-green-700' : 'text-yellow-700'
                  }`}>
                    {testSummary.passed} of {testSummary.total} tests passed ({testSummary.percentage}%)
                  </p>
                </div>
                <div className={`text-3xl ${
                  testSummary.allPassed ? 'text-green-600' : 'text-yellow-600'
                }`}>
                  {testSummary.allPassed ? '✅' : '⚠️'}
                </div>
              </div>
            </div>
          )}

          {/* Individual test results */}
          <div className="space-y-2">
            {testResults.map((result, index) => (
              <div
                key={index}
                className={`p-3 rounded-lg border ${
                  result.passed
                    ? 'bg-green-50 border-green-200'
                    : 'bg-red-50 border-red-200'
                }`}
              >
                <div className="flex items-start gap-2">
                  <span className={`text-lg ${
                    result.passed ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {result.passed ? '✓' : '✗'}
                  </span>
                  <div className="flex-1">
                    <p className={`text-sm font-medium ${
                      result.passed ? 'text-green-900' : 'text-red-900'
                    }`}>
                      {result.description}
                    </p>
                    {result.error && (
                      <p className="text-xs text-red-700 mt-1">
                        Error: {result.error}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Code explanation (review mode) */}
      {showExplanation && question.explanation && (
        <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
          <h4 className="font-semibold text-gray-900 mb-2">Expected Solution:</h4>
          <pre className="text-sm text-gray-700 whitespace-pre-wrap font-mono">
            {typeof question.correctAnswer === 'object'
              ? JSON.stringify(question.correctAnswer, null, 2)
              : question.correctAnswer}
          </pre>
        </div>
      )}
    </div>
  );
};

export default CodeChallengeQuestion;
