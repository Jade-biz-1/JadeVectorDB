import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { CheckCircle2, XCircle, Code, Bug, Lightbulb } from 'lucide-react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

/**
 * QuizQuestion Component
 * Renders different types of quiz questions with appropriate UI
 */
const QuizQuestion = ({
  question,
  questionNumber,
  totalQuestions,
  userAnswer,
  onAnswerChange,
  onNext,
  onPrevious,
  onSubmit,
  showResults = false,
  isCorrect = false,
  isFirstQuestion,
  isLastQuestion
}) => {
  const [localAnswer, setLocalAnswer] = useState(userAnswer || '');

  useEffect(() => {
    setLocalAnswer(userAnswer || '');
  }, [userAnswer, question.id]);

  const handleAnswerChange = (value) => {
    setLocalAnswer(value);
    onAnswerChange(value);
  };

  const getQuestionIcon = () => {
    switch (question.type) {
      case 'code-completion':
        return <Code className="w-5 h-5" />;
      case 'debugging':
        return <Bug className="w-5 h-5" />;
      case 'scenario-based':
        return <Lightbulb className="w-5 h-5" />;
      default:
        return null;
    }
  };

  const getDifficultyColor = () => {
    switch (question.difficulty) {
      case 'easy':
        return 'text-green-600 dark:text-green-400';
      case 'medium':
        return 'text-yellow-600 dark:text-yellow-400';
      case 'hard':
        return 'text-red-600 dark:text-red-400';
      default:
        return 'text-gray-600 dark:text-gray-400';
    }
  };

  const renderMultipleChoice = () => (
    <RadioGroup
      value={localAnswer.toString()}
      onValueChange={(value) => handleAnswerChange(parseInt(value))}
      disabled={showResults}
    >
      <div className="space-y-3">
        {question.options.map((option, index) => (
          <div
            key={index}
            className={`flex items-start space-x-3 p-4 rounded-lg border-2 transition-all ${
              showResults
                ? index === question.correctAnswer
                  ? 'border-green-500 bg-green-50 dark:bg-green-950'
                  : localAnswer === index
                  ? 'border-red-500 bg-red-50 dark:bg-red-950'
                  : 'border-gray-200 dark:border-gray-700'
                : localAnswer === index
                ? 'border-blue-500 bg-blue-50 dark:bg-blue-950'
                : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
            }`}
          >
            <RadioGroupItem value={index.toString()} id={`option-${index}`} />
            <Label
              htmlFor={`option-${index}`}
              className="flex-1 cursor-pointer leading-relaxed"
            >
              {option}
            </Label>
            {showResults && index === question.correctAnswer && (
              <CheckCircle2 className="w-5 h-5 text-green-600 flex-shrink-0" />
            )}
            {showResults && localAnswer === index && index !== question.correctAnswer && (
              <XCircle className="w-5 h-5 text-red-600 flex-shrink-0" />
            )}
          </div>
        ))}
      </div>
    </RadioGroup>
  );

  const renderCodeCompletion = () => (
    <div className="space-y-4">
      <div className="bg-gray-900 rounded-lg overflow-hidden">
        <div className="bg-gray-800 px-4 py-2 text-sm text-gray-300 font-mono">
          Complete the code below:
        </div>
        <SyntaxHighlighter
          language="python"
          style={vscDarkPlus}
          customStyle={{ margin: 0, padding: '1rem' }}
        >
          {question.codeTemplate}
        </SyntaxHighlighter>
      </div>

      <div>
        <Label htmlFor="code-answer" className="text-sm font-medium mb-2 block">
          Your Answer:
        </Label>
        <Textarea
          id="code-answer"
          value={localAnswer}
          onChange={(e) => handleAnswerChange(e.target.value)}
          placeholder="Type your code here..."
          className="font-mono text-sm min-h-[150px]"
          disabled={showResults}
        />
      </div>

      {showResults && (
        <div className="space-y-3">
          <Alert className={isCorrect ? 'border-green-500' : 'border-red-500'}>
            <AlertDescription>
              <div className="font-semibold mb-2">
                {isCorrect ? '✅ Correct!' : '❌ Incorrect'}
              </div>
              <div className="text-sm">Expected answer:</div>
              <div className="mt-2 bg-gray-900 rounded p-3">
                <code className="text-green-400 text-sm">{question.correctAnswer}</code>
              </div>
            </AlertDescription>
          </Alert>
        </div>
      )}
    </div>
  );

  const renderDebugging = () => (
    <div className="space-y-4">
      <div className="bg-gray-900 rounded-lg overflow-hidden">
        <div className="bg-red-900 px-4 py-2 text-sm text-red-100 font-mono flex items-center gap-2">
          <Bug className="w-4 h-4" />
          Find and fix the bug in this code:
        </div>
        <SyntaxHighlighter
          language="python"
          style={vscDarkPlus}
          customStyle={{ margin: 0, padding: '1rem' }}
          showLineNumbers
        >
          {question.buggyCode}
        </SyntaxHighlighter>
      </div>

      {question.errorMessage && (
        <Alert variant="destructive">
          <AlertDescription className="font-mono text-sm">
            {question.errorMessage}
          </AlertDescription>
        </Alert>
      )}

      <div>
        <Label htmlFor="debug-answer" className="text-sm font-medium mb-2 block">
          Your Fixed Code:
        </Label>
        <Textarea
          id="debug-answer"
          value={localAnswer}
          onChange={(e) => handleAnswerChange(e.target.value)}
          placeholder="Paste the corrected code here..."
          className="font-mono text-sm min-h-[150px]"
          disabled={showResults}
        />
      </div>

      {showResults && (
        <Alert className={isCorrect ? 'border-green-500' : 'border-yellow-500'}>
          <AlertDescription>
            <div className="font-semibold mb-2">
              {isCorrect ? '✅ Correct!' : '💡 Review the fix'}
            </div>
            <div className="text-sm mb-2">The bug was:</div>
            <div className="bg-gray-900 rounded p-3">
              <code className="text-yellow-400 text-sm">{question.correctAnswer}</code>
            </div>
          </AlertDescription>
        </Alert>
      )}
    </div>
  );

  const renderScenarioBased = () => (
    <div className="space-y-4">
      <div className="bg-blue-50 dark:bg-blue-950 border-l-4 border-blue-500 p-4 rounded">
        <div className="flex items-start gap-3">
          <Lightbulb className="w-5 h-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-1" />
          <div>
            <h4 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">
              Scenario
            </h4>
            <p className="text-blue-800 dark:text-blue-200 text-sm leading-relaxed">
              {question.scenario}
            </p>
          </div>
        </div>
      </div>

      <RadioGroup
        value={localAnswer.toString()}
        onValueChange={(value) => handleAnswerChange(parseInt(value))}
        disabled={showResults}
      >
        <div className="space-y-3">
          {question.options.map((option, index) => (
            <div
              key={index}
              className={`flex items-start space-x-3 p-4 rounded-lg border-2 transition-all ${
                showResults
                  ? index === question.correctAnswer
                    ? 'border-green-500 bg-green-50 dark:bg-green-950'
                    : localAnswer === index
                    ? 'border-red-500 bg-red-50 dark:bg-red-950'
                    : 'border-gray-200 dark:border-gray-700'
                  : localAnswer === index
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-950'
                  : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
              }`}
            >
              <RadioGroupItem value={index.toString()} id={`scenario-${index}`} />
              <Label
                htmlFor={`scenario-${index}`}
                className="flex-1 cursor-pointer leading-relaxed"
              >
                {option}
              </Label>
              {showResults && index === question.correctAnswer && (
                <CheckCircle2 className="w-5 h-5 text-green-600 flex-shrink-0" />
              )}
            </div>
          ))}
        </div>
      </RadioGroup>
    </div>
  );

  const renderQuestion = () => {
    switch (question.type) {
      case 'multiple-choice':
        return renderMultipleChoice();
      case 'code-completion':
        return renderCodeCompletion();
      case 'debugging':
        return renderDebugging();
      case 'scenario-based':
        return renderScenarioBased();
      default:
        return <div>Unknown question type</div>;
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-gray-500 dark:text-gray-400">
              Question {questionNumber} of {totalQuestions}
            </span>
            {getQuestionIcon()}
          </div>
          <div className="flex items-center gap-3">
            <span className={`text-sm font-medium capitalize ${getDifficultyColor()}`}>
              {question.difficulty}
            </span>
            <span className="text-sm text-gray-500 dark:text-gray-400">
              {question.points} {question.points === 1 ? 'point' : 'points'}
            </span>
          </div>
        </div>
        <CardTitle className="text-xl leading-relaxed">{question.question}</CardTitle>
        {question.hint && !showResults && (
          <CardDescription className="mt-2 text-sm italic">
            💡 Hint: {question.hint}
          </CardDescription>
        )}
      </CardHeader>

      <CardContent className="space-y-6">
        {renderQuestion()}

        {showResults && question.explanation && (
          <Alert className="border-blue-500 bg-blue-50 dark:bg-blue-950">
            <AlertDescription>
              <div className="font-semibold text-blue-900 dark:text-blue-100 mb-2">
                📚 Explanation
              </div>
              <p className="text-blue-800 dark:text-blue-200 text-sm leading-relaxed">
                {question.explanation}
              </p>
            </AlertDescription>
          </Alert>
        )}

        <div className="flex justify-between pt-4 border-t">
          <Button
            variant="outline"
            onClick={onPrevious}
            disabled={isFirstQuestion}
          >
            ← Previous
          </Button>

          <div className="flex gap-2">
            {!isLastQuestion ? (
              <Button onClick={onNext}>
                Next →
              </Button>
            ) : (
              !showResults && (
                <Button onClick={onSubmit} variant="default">
                  Submit Quiz
                </Button>
              )
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default QuizQuestion;
