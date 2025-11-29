import React, { useState, useEffect } from 'react';
import Quiz from './Quiz';
import QuizResults from './QuizResults';
import assessmentState from '../../lib/assessmentState';
import { gradeQuestion, calculateTotalScore, isPassing, analyzePerformance, calculateTimeMetrics } from '../../lib/quizScoring';

/**
 * AssessmentSystem - Main container for quiz assessments
 *
 * Manages the overall assessment flow including quiz presentation,
 * grading, and results display.
 */
const AssessmentSystem = ({
  moduleId,
  onComplete,
  onRetry,
  minPassScore = 70
}) => {
  const [quizData, setQuizData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showResults, setShowResults] = useState(false);
  const [results, setResults] = useState(null);
  const [session, setSession] = useState(null);

  // Load quiz data for the module
  useEffect(() => {
    const loadQuizData = async () => {
      try {
        setLoading(true);

        // Import quiz data dynamically
        const quizModule = await import(`../../data/quizzes/${moduleId}_quiz.json`);
        const data = quizModule.default;

        setQuizData(data);

        // Initialize assessment session
        const newSession = assessmentState.initAssessment(moduleId, data);
        setSession(newSession);

        setLoading(false);
      } catch (err) {
        console.error('Failed to load quiz data:', err);
        setError('Failed to load assessment. Please try again.');
        setLoading(false);
      }
    };

    loadQuizData();
  }, [moduleId]);

  /**
   * Handle quiz submission
   */
  const handleQuizSubmit = (answers) => {
    if (!quizData || !session) return;

    // Grade all questions
    const gradedResults = quizData.questions.map(question => {
      const userAnswer = answers[question.id];
      return {
        ...gradeQuestion(question, userAnswer),
        difficulty: question.difficulty
      };
    });

    // Calculate total score
    const scoreData = calculateTotalScore(gradedResults);

    // Check if passing
    const passed = isPassing(scoreData.percentage, minPassScore);

    // Analyze performance
    const analysis = analyzePerformance(gradedResults);

    // Calculate time metrics
    const timeMetrics = calculateTimeMetrics(session.startTime, Date.now());

    // Create result object
    const result = {
      moduleId,
      moduleName: quizData.moduleName,
      score: scoreData.percentage,
      passed,
      totalPoints: scoreData.totalPoints,
      earnedPoints: scoreData.earnedPoints,
      correctCount: scoreData.correctCount,
      totalQuestions: scoreData.totalQuestions,
      timeSpent: timeMetrics.totalMs,
      timeFormatted: timeMetrics.formatted,
      gradedResults,
      analysis,
      passingScore: minPassScore,
      attemptNumber: assessmentState.getModuleHistory(moduleId).length + 1
    };

    // Complete assessment in state manager
    assessmentState.completeAssessment(result);

    // Update state
    setResults(result);
    setShowResults(true);

    // Notify parent component
    if (onComplete) {
      onComplete(result);
    }
  };

  /**
   * Handle retry action
   */
  const handleRetry = () => {
    // Reset states
    setShowResults(false);
    setResults(null);

    // Reinitialize session
    const newSession = assessmentState.initAssessment(moduleId, quizData);
    setSession(newSession);

    // Notify parent if callback provided
    if (onRetry) {
      onRetry();
    }
  };

  /**
   * Handle answer change
   */
  const handleAnswerChange = (questionId, answer) => {
    assessmentState.saveAnswer(questionId, answer);
  };

  // Loading state
  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading assessment...</p>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="text-red-500 text-5xl mb-4">⚠️</div>
          <h3 className="text-xl font-semibold text-gray-800 mb-2">Error Loading Assessment</h3>
          <p className="text-gray-600 mb-4">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  // Show results
  if (showResults && results) {
    return (
      <QuizResults
        results={results}
        onRetry={handleRetry}
        onContinue={onComplete}
      />
    );
  }

  // Show quiz
  return (
    <div className="max-w-4xl mx-auto p-6">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-3xl font-bold text-gray-800 mb-2">
          {quizData.moduleName} - Assessment
        </h2>
        <p className="text-gray-600">
          Test your knowledge of this module. You need {minPassScore}% to pass.
        </p>

        {/* Previous attempts info */}
        {assessmentState.getModuleHistory(moduleId).length > 0 && (
          <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-sm text-blue-800">
              📊 Previous attempts: {assessmentState.getModuleHistory(moduleId).length}
              {' | '}
              Best score: {assessmentState.getBestScore(moduleId)}%
              {assessmentState.hasPassedModule(moduleId) && ' ✓ Passed'}
            </p>
          </div>
        )}
      </div>

      {/* Quiz */}
      <Quiz
        questions={quizData.questions}
        onSubmit={handleQuizSubmit}
        onAnswerChange={handleAnswerChange}
        timeLimit={quizData.timeLimit}
      />
    </div>
  );
};

export default AssessmentSystem;
