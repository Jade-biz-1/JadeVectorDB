import React, { useState } from 'react';
import { getGradeLetter, getPerformanceLevel } from '../../lib/quizScoring';

/**
 * QuizResults - Display quiz results and performance analysis
 *
 * Shows score, performance breakdown, and options to retry or continue.
 */
const QuizResults = ({
  results,
  onRetry,
  onContinue
}) => {
  const [showDetailedResults, setShowDetailedResults] = useState(false);
  const [showAnalysis, setShowAnalysis] = useState(true);

  const performanceLevel = getPerformanceLevel(results.score);
  const gradeLetter = getGradeLetter(results.score);

  /**
   * Render score circle with animation
   */
  const renderScoreCircle = () => {
    const circumference = 2 * Math.PI * 70; // radius = 70
    const strokeDashoffset = circumference - (results.score / 100) * circumference;

    return (
      <div className="relative w-48 h-48">
        <svg className="transform -rotate-90 w-48 h-48">
          {/* Background circle */}
          <circle
            cx="96"
            cy="96"
            r="70"
            stroke="#e5e7eb"
            strokeWidth="12"
            fill="none"
          />
          {/* Progress circle */}
          <circle
            cx="96"
            cy="96"
            r="70"
            stroke={performanceLevel.color}
            strokeWidth="12"
            fill="none"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            className="transition-all duration-1000 ease-out"
          />
        </svg>
        {/* Score text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-4xl font-bold" style={{ color: performanceLevel.color }}>
            {results.score}%
          </span>
          <span className="text-lg font-semibold text-gray-600 mt-1">
            Grade {gradeLetter}
          </span>
        </div>
      </div>
    );
  };

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-3xl font-bold text-gray-800 mb-2">
          Assessment Complete!
        </h2>
        <p className="text-gray-600">
          {results.moduleName}
        </p>
      </div>

      {/* Pass/Fail Banner */}
      <div className={`p-6 rounded-lg text-center ${
        results.passed
          ? 'bg-green-50 border-2 border-green-300'
          : 'bg-orange-50 border-2 border-orange-300'
      }`}>
        <div className="text-6xl mb-3">
          {results.passed ? '🎉' : '📚'}
        </div>
        <h3 className={`text-2xl font-bold mb-2 ${
          results.passed ? 'text-green-900' : 'text-orange-900'
        }`}>
          {results.passed ? 'Congratulations! You Passed!' : 'Keep Learning!'}
        </h3>
        <p className={`text-lg ${
          results.passed ? 'text-green-700' : 'text-orange-700'
        }`}>
          {results.passed
            ? 'You have successfully completed this module assessment.'
            : `You need ${results.passingScore}% to pass. Review the material and try again.`}
        </p>
      </div>

      {/* Score Overview */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex flex-col md:flex-row items-center justify-around gap-8">
          {/* Score circle */}
          <div className="flex flex-col items-center">
            {renderScoreCircle()}
            <div className="mt-4 text-center">
              <p className="font-semibold" style={{ color: performanceLevel.color }}>
                {performanceLevel.level}
              </p>
              <p className="text-sm text-gray-600">
                {performanceLevel.description}
              </p>
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 gap-6">
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-600">
                {results.correctCount}
              </div>
              <div className="text-sm text-gray-600">Correct</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-gray-600">
                {results.totalQuestions - results.correctCount}
              </div>
              <div className="text-sm text-gray-600">Incorrect</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-green-600">
                {results.earnedPoints}
              </div>
              <div className="text-sm text-gray-600">Points Earned</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-gray-600">
                {results.totalPoints}
              </div>
              <div className="text-sm text-gray-600">Total Points</div>
            </div>
          </div>
        </div>

        {/* Additional info */}
        <div className="mt-6 pt-6 border-t border-gray-200 flex justify-around text-sm text-gray-600">
          <div>
            <span className="font-semibold">Time:</span> {results.timeFormatted}
          </div>
          <div>
            <span className="font-semibold">Attempt:</span> #{results.attemptNumber}
          </div>
          <div>
            <span className="font-semibold">Passing Score:</span> {results.passingScore}%
          </div>
        </div>
      </div>

      {/* Performance Analysis */}
      {showAnalysis && results.analysis && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-xl font-bold text-gray-800">Performance Analysis</h3>
            <button
              onClick={() => setShowAnalysis(false)}
              className="text-sm text-gray-500 hover:text-gray-700"
            >
              Hide
            </button>
          </div>

          {/* By Difficulty */}
          <div className="space-y-4">
            <div>
              <h4 className="font-semibold text-gray-700 mb-3">By Difficulty</h4>
              <div className="space-y-2">
                {Object.entries(results.analysis.byDifficulty).map(([difficulty, data]) => {
                  if (data.total === 0) return null;
                  return (
                    <div key={difficulty} className="flex items-center gap-4">
                      <div className="w-24 text-sm text-gray-600 capitalize">{difficulty}</div>
                      <div className="flex-1">
                        <div className="w-full bg-gray-200 rounded-full h-4">
                          <div
                            className={`h-4 rounded-full ${
                              data.percentage >= 80 ? 'bg-green-500' :
                              data.percentage >= 60 ? 'bg-yellow-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${data.percentage}%` }}
                          />
                        </div>
                      </div>
                      <div className="w-24 text-sm text-gray-700 text-right">
                        {data.correct}/{data.total} ({data.percentage}%)
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Strengths */}
            {results.analysis.strengths.length > 0 && (
              <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                <h4 className="font-semibold text-green-900 mb-2 flex items-center gap-2">
                  <span>💪</span> Strengths
                </h4>
                <ul className="list-disc list-inside text-sm text-green-700 space-y-1">
                  {results.analysis.strengths.map((strength, index) => (
                    <li key={index}>{strength}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Weaknesses */}
            {results.analysis.weaknesses.length > 0 && (
              <div className="p-4 bg-orange-50 border border-orange-200 rounded-lg">
                <h4 className="font-semibold text-orange-900 mb-2 flex items-center gap-2">
                  <span>📖</span> Areas for Improvement
                </h4>
                <ul className="list-disc list-inside text-sm text-orange-700 space-y-1">
                  {results.analysis.weaknesses.map((weakness, index) => (
                    <li key={index}>{weakness}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Recommendations */}
            {results.analysis.recommendations.length > 0 && (
              <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <h4 className="font-semibold text-blue-900 mb-2 flex items-center gap-2">
                  <span>💡</span> Recommendations
                </h4>
                <ul className="list-disc list-inside text-sm text-blue-700 space-y-1">
                  {results.analysis.recommendations.map((rec, index) => (
                    <li key={index}>{rec}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Detailed Results */}
      {showDetailedResults && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-xl font-bold text-gray-800">Detailed Results</h3>
            <button
              onClick={() => setShowDetailedResults(false)}
              className="text-sm text-gray-500 hover:text-gray-700"
            >
              Hide
            </button>
          </div>

          <div className="space-y-4">
            {results.gradedResults.map((result, index) => (
              <div
                key={result.questionId}
                className={`p-4 rounded-lg border-2 ${
                  result.isCorrect
                    ? 'bg-green-50 border-green-300'
                    : 'bg-red-50 border-red-300'
                }`}
              >
                <div className="flex items-start gap-3">
                  <span className={`text-2xl ${
                    result.isCorrect ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {result.isCorrect ? '✓' : '✗'}
                  </span>
                  <div className="flex-1">
                    <div className="flex justify-between items-start mb-2">
                      <h4 className="font-semibold text-gray-800">
                        Question {index + 1}
                      </h4>
                      <span className={`text-sm font-semibold ${
                        result.isCorrect ? 'text-green-700' : 'text-red-700'
                      }`}>
                        {result.earnedPoints}/{result.points} pts
                        {result.partialCredit && ' (Partial Credit)'}
                      </span>
                    </div>
                    {result.explanation && (
                      <p className="text-sm text-gray-700 mt-2">
                        {result.explanation}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Toggle detailed results button */}
      {!showDetailedResults && (
        <div className="text-center">
          <button
            onClick={() => setShowDetailedResults(true)}
            className="text-blue-600 hover:text-blue-700 font-medium"
          >
            📊 Show Detailed Results
          </button>
        </div>
      )}

      {/* Action buttons */}
      <div className="flex justify-center gap-4">
        {!results.passed && onRetry && (
          <button
            onClick={onRetry}
            className="px-8 py-3 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 shadow-lg"
          >
            🔄 Retry Assessment
          </button>
        )}

        {results.passed && onContinue && (
          <button
            onClick={() => onContinue(results)}
            className="px-8 py-3 bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700 shadow-lg"
          >
            Continue to Next Module →
          </button>
        )}

        {results.passed && (
          <button
            onClick={onRetry}
            className="px-8 py-3 bg-gray-600 text-white rounded-lg font-semibold hover:bg-gray-700"
          >
            Retake to Improve Score
          </button>
        )}
      </div>
    </div>
  );
};

export default QuizResults;
