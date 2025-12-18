# Assessment System Documentation

## Overview

The Assessment System provides a comprehensive quiz and testing framework for the JadeVectorDB tutorial. It includes support for multiple question types, automatic grading, performance analysis, and progress tracking.

## Architecture

```
Assessment System
├── State Management
│   ├── assessmentState.js       - Session and history management
│   └── quizScoring.js          - Grading and analysis logic
├── Data
│   └── quizzes/                - Quiz JSON files for each module
│       ├── module1_quiz.json
│       ├── module2_quiz.json
│       └── ...
└── Components
    ├── AssessmentSystem.jsx    - Main container
    ├── Quiz.jsx                - Quiz presentation
    ├── QuizProgress.jsx        - Progress indicator
    ├── QuizQuestion.jsx        - Question router
    ├── MultipleChoiceQuestion.jsx
    ├── TrueFalseQuestion.jsx
    ├── CodeChallengeQuestion.jsx
    └── QuizResults.jsx         - Results display
```

## Features

### Question Types

1. **Multiple Choice**
   - Single answer
   - Multiple answers (checkbox style)
   - Visual selection feedback
   - Review mode with correct/incorrect indicators

2. **True/False**
   - Large, accessible buttons
   - Clear visual feedback
   - Binary choice interface

3. **Code Challenges**
   - Code editor with syntax highlighting
   - Test case validation
   - Partial credit support
   - Real-time feedback

4. **Fill in the Blank**
   - Text input
   - Case-insensitive matching
   - Trimmed comparison

### Key Features

- **Progressive Hints**: 3-level hint system (subtle → moderate → explicit)
- **Time Limits**: Optional timed assessments
- **Question Navigation**: Jump to any question, review before submit
- **Partial Credit**: For code challenges with multiple test cases
- **Performance Analysis**: Breakdown by difficulty and question type
- **Attempt Tracking**: History of all attempts with scores
- **Module Unlocking**: Automatically unlock next module on pass
- **Responsive Design**: Works on desktop and mobile
- **Accessibility**: Keyboard navigation and screen reader support

## Quick Start

### 1. Import the AssessmentSystem Component

```jsx
import AssessmentSystem from '@/components/tutorial/AssessmentSystem';
```

### 2. Add to Your Tutorial Module

```jsx
import React, { useState } from 'react';
import AssessmentSystem from '@/components/tutorial/AssessmentSystem';

const MyTutorialModule = () => {
  const [showAssessment, setShowAssessment] = useState(false);
  const [moduleComplete, setModuleComplete] = useState(false);

  const handleModuleComplete = () => {
    setShowAssessment(true);
  };

  const handleAssessmentComplete = (result) => {
    if (result.passed) {
      setModuleComplete(true);
      // Unlock next module, show success message, etc.
    }
  };

  const handleRetry = () => {
    setShowAssessment(false);
    // Reset to tutorial beginning if desired
  };

  if (showAssessment) {
    return (
      <AssessmentSystem
        moduleId="module1"
        onComplete={handleAssessmentComplete}
        onRetry={handleRetry}
        minPassScore={70}
      />
    );
  }

  return (
    <div>
      {/* Your tutorial content */}
      <button onClick={handleModuleComplete}>
        Take Assessment
      </button>
    </div>
  );
};

export default MyTutorialModule;
```

### 3. Create Quiz Data

Create a JSON file in `tutorial/src/data/quizzes/` following this schema:

```json
{
  "moduleId": "module1",
  "moduleName": "Getting Started",
  "version": "1.0",
  "passingScore": 70,
  "timeLimit": null,
  "questions": [
    {
      "id": "m1_q1",
      "type": "multiple-choice",
      "question": "What is a vector database?",
      "points": 10,
      "difficulty": "easy",
      "options": [
        "Option A",
        "Option B",
        "Option C",
        "Option D"
      ],
      "correctAnswer": 0,
      "explanation": "Detailed explanation...",
      "hints": [
        "Subtle hint",
        "Moderate hint",
        "Explicit hint"
      ]
    }
  ]
}
```

## Component API

### AssessmentSystem

Main container component that manages the assessment flow.

**Props:**

| Prop | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `moduleId` | string | Yes | - | Module identifier (e.g., 'module1') |
| `onComplete` | function | No | - | Callback when assessment completes |
| `onRetry` | function | No | - | Callback when user retries |
| `minPassScore` | number | No | 70 | Minimum score to pass (0-100) |

**onComplete Callback:**

Receives a result object:

```javascript
{
  moduleId: 'module1',
  moduleName: 'Getting Started',
  score: 85,                    // Percentage (0-100)
  passed: true,                 // Whether passed
  totalPoints: 100,
  earnedPoints: 85,
  correctCount: 7,
  totalQuestions: 8,
  timeSpent: 180000,            // Milliseconds
  timeFormatted: '3:00',
  gradedResults: [...],         // Detailed results
  analysis: {...},              // Performance analysis
  passingScore: 70,
  attemptNumber: 1
}
```

### Quiz

Quiz presentation component (used internally by AssessmentSystem).

**Props:**

| Prop | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `questions` | array | Yes | - | Array of question objects |
| `onSubmit` | function | Yes | - | Submit handler |
| `onAnswerChange` | function | No | - | Answer change callback |
| `allowNavigation` | boolean | No | true | Allow jumping between questions |
| `timeLimit` | number | No | null | Time limit in seconds |

### QuizQuestion

Question router component (used internally by Quiz).

**Props:**

| Prop | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `question` | object | Yes | - | Question object |
| `answer` | any | No | - | Current answer |
| `onChange` | function | Yes | - | Answer change handler |
| `questionNumber` | number | Yes | - | Display number (1-indexed) |
| `totalQuestions` | number | Yes | - | Total question count |
| `readOnly` | boolean | No | false | Disable editing |
| `showExplanation` | boolean | No | false | Show explanation |

## Quiz Data Schema

### Question Object

```typescript
{
  id: string;                    // Unique identifier
  type: 'multiple-choice' | 'true-false' | 'code-challenge' | 'fill-blank';
  question: string;              // Question text
  points: number;                // Point value
  difficulty: 'easy' | 'medium' | 'hard';

  // Multiple choice specific
  options?: string[];            // Answer options
  multipleAnswers?: boolean;     // Allow multiple selections
  correctAnswer: number | number[] | boolean | string | object;

  // Code challenge specific
  codeTemplate?: string;         // Starting code
  testCases?: TestCase[];        // Validation tests

  // Common
  explanation: string;           // Answer explanation
  hints?: string[];              // Progressive hints (3 max)
}
```

### Test Case Object

```typescript
{
  description: string;           // Test description
  validate: string;              // JavaScript expression to evaluate
}
```

**Example:**

```json
{
  "description": "Database name should be 'my-db'",
  "validate": "config.name === 'my-db'"
}
```

The `validate` expression has access to these context variables:
- `config` - User's answer (for configuration objects)
- `params` - User's answer (for parameter objects)
- `vector` - User's answer (for vector objects)
- `update` - User's answer (for update objects)

## State Management

### Assessment State

Access assessment state programmatically:

```javascript
import assessmentState from '@/lib/assessmentState';

// Get module history
const history = assessmentState.getModuleHistory('module1');

// Get best score
const bestScore = assessmentState.getBestScore('module1');

// Check if passed
const passed = assessmentState.hasPassedModule('module1');

// Get overall progress
const progress = assessmentState.getOverallProgress();
// Returns: { totalModules, completedModules, passedModules, totalAttempts, averageScore, bestScores }

// Get module statistics
const stats = assessmentState.getModuleStatistics('module1');
// Returns: { attempts, passed, bestScore, averageScore, totalTimeSpent, lastAttempt, improvement }
```

### Quiz Scoring

Use scoring utilities:

```javascript
import {
  gradeQuestion,
  calculateTotalScore,
  isPassing,
  analyzePerformance,
  getGradeLetter,
  getPerformanceLevel
} from '@/lib/quizScoring';

// Grade a single question
const result = gradeQuestion(question, userAnswer);

// Calculate total score
const scoreData = calculateTotalScore(gradedResults);

// Check if passing
const passed = isPassing(85, 70); // (score, minScore)

// Analyze performance
const analysis = analyzePerformance(gradedResults);

// Get grade letter
const letter = getGradeLetter(85); // Returns 'A', 'B', 'C', 'D', or 'F'

// Get performance level
const level = getPerformanceLevel(85);
// Returns: { level, description, color }
```

## Styling

The assessment system uses Tailwind CSS classes. Key color schemes:

- **Primary (Blue)**: Navigation, selected states
- **Success (Green)**: Correct answers, passed status
- **Danger (Red)**: Incorrect answers, failed tests
- **Warning (Yellow)**: Hints, partial credit
- **Info (Blue)**: Information, explanations

### Customization

Override styles using Tailwind's utility classes or custom CSS:

```css
/* Custom styles for assessment components */
.assessment-container {
  /* Your custom styles */
}
```

## Best Practices

### Quiz Design

1. **Question Balance**
   - Mix difficulty levels (40% easy, 40% medium, 20% hard)
   - Vary question types for engagement
   - 8-12 questions per module is optimal

2. **Clear Questions**
   - Use concise, unambiguous language
   - Avoid trick questions
   - Provide context when needed

3. **Meaningful Feedback**
   - Write detailed explanations
   - Explain why wrong answers are incorrect
   - Provide learning resources in hints

4. **Fair Grading**
   - Set appropriate point values (harder = more points)
   - Use partial credit for complex questions
   - Set reasonable passing scores (70% recommended)

### Integration

1. **Progressive Disclosure**
   - Show assessment after tutorial completion
   - Don't make assessment mandatory for exploration
   - Allow unlimited retries

2. **Motivation**
   - Celebrate success with animations
   - Show progress and improvement
   - Provide constructive feedback on failure

3. **Accessibility**
   - Ensure keyboard navigation works
   - Provide sufficient color contrast
   - Use semantic HTML
   - Add ARIA labels where needed

## Examples

### Multiple Choice Question

```json
{
  "id": "q1",
  "type": "multiple-choice",
  "question": "Which metrics does JadeVectorDB support?",
  "points": 10,
  "difficulty": "easy",
  "multipleAnswers": true,
  "options": [
    "Cosine similarity",
    "Euclidean distance",
    "Dot product",
    "Hamming distance"
  ],
  "correctAnswer": [0, 1, 2],
  "explanation": "JadeVectorDB supports cosine similarity, Euclidean distance, and dot product. Hamming distance is for binary data.",
  "hints": [
    "Think about vector similarity metrics",
    "Three of these are commonly used for vectors",
    "Hamming distance is for strings, not vectors"
  ]
}
```

### Code Challenge Question

```json
{
  "id": "q2",
  "type": "code-challenge",
  "question": "Create a database configuration with name 'my-db' and dimension 384",
  "points": 15,
  "difficulty": "medium",
  "codeTemplate": "const config = {\n  name: '',\n  dimension: 0\n};",
  "correctAnswer": {
    "name": "my-db",
    "dimension": 384
  },
  "testCases": [
    {
      "description": "Name should be 'my-db'",
      "validate": "config.name === 'my-db'"
    },
    {
      "description": "Dimension should be 384",
      "validate": "config.dimension === 384"
    }
  ],
  "explanation": "The configuration object needs both name and dimension properties set correctly.",
  "hints": [
    "Check your property names",
    "Make sure dimension is a number, not a string",
    "The name should be exactly 'my-db'"
  ]
}
```

### True/False Question

```json
{
  "id": "q3",
  "type": "true-false",
  "question": "All vectors in a database must have the same dimension.",
  "points": 10,
  "difficulty": "easy",
  "correctAnswer": true,
  "explanation": "Yes, all vectors in a database must have the same dimension to enable meaningful similarity calculations.",
  "hints": [
    "Think about how similarity is calculated",
    "Can you compare vectors of different sizes?"
  ]
}
```

## Troubleshooting

### Common Issues

**Quiz not loading:**
- Check that quiz JSON file exists in `tutorial/src/data/quizzes/`
- Verify moduleId matches filename (e.g., 'module1' → 'module1_quiz.json')
- Check JSON syntax for errors

**Answers not saving:**
- Ensure onChange callback is provided
- Check browser console for localStorage errors
- Verify assessmentState is imported correctly

**Test cases failing:**
- Use `console.log` to debug validate expressions
- Check that context variables match your question type
- Ensure user answer format matches expected format

**Styling issues:**
- Verify Tailwind CSS is configured
- Check for CSS conflicts
- Use browser dev tools to inspect elements

## Future Enhancements

Planned features:

- [ ] Timed individual questions
- [ ] Drag-and-drop questions
- [ ] Image-based questions
- [ ] Math equation rendering
- [ ] Code execution sandbox
- [ ] Adaptive difficulty
- [ ] Peer comparison analytics
- [ ] Certificate generation
- [ ] Export results to PDF

## Support

For questions or issues:
- Check this documentation
- Review example quiz files
- Examine component source code
- Test with browser dev tools

## License

Part of JadeVectorDB tutorial system.
