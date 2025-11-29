# Integration Guide

This guide explains how to integrate the Assessment System, Achievement System, Contextual Help System, and Readiness Assessment into your tutorial modules.

## Table of Contents

1. [Quick Start](#quick-start)
2. [TutorialWrapper Component](#tutorialwrapper-component)
3. [Adding Assessments](#adding-assessments)
4. [Achievement Integration](#achievement-integration)
5. [Contextual Help](#contextual-help)
6. [Readiness Assessment](#readiness-assessment)
7. [Complete Example](#complete-example)

---

## Quick Start

### Using TutorialWrapper (Recommended)

The easiest way to integrate all systems is to use the `TutorialWrapper` component:

```jsx
import TutorialWrapper from './components/tutorial/TutorialWrapper';

const MyModule = () => {
  return (
    <TutorialWrapper
      moduleId="module1"
      moduleName="Getting Started"
      onComplete={() => navigate('/next-module')}
    >
      {/* Your module content here */}
      <div>
        <h2>Module Content</h2>
        <p>Your tutorial steps...</p>
      </div>
    </TutorialWrapper>
  );
};
```

This automatically provides:
- Assessment system integration
- Achievement tracking
- Help system with F1 shortcut
- Module completion status
- Readiness assessment trigger (when all modules complete)

---

## TutorialWrapper Component

### Props

```typescript
interface TutorialWrapperProps {
  moduleId: string;              // Unique module identifier (e.g., "module1")
  moduleName: string;             // Display name (e.g., "Getting Started")
  children: React.ReactNode;      // Your module content
  onComplete?: () => void;        // Callback when module is completed
  showReadinessWhenComplete?: boolean; // Show readiness assessment after all modules
}
```

### Features

1. **Assessment Trigger**: Displays "Take Assessment" button after module content
2. **Achievement Notifications**: Automatically shows achievement unlocks
3. **Help Button**: Fixed help button (bottom-right) + F1 keyboard shortcut
4. **Completion Status**: Shows completion banner when module is passed
5. **Retry Support**: Allows retaking assessments to improve scores

---

## Adding Assessments

### Step 1: Create Quiz Data

Create a JSON file in `src/data/quizzes/`:

```json
{
  "moduleId": "module1",
  "moduleName": "Getting Started",
  "passingScore": 70,
  "questions": [
    {
      "id": "m1_q1",
      "type": "multiple-choice",
      "question": "What is a vector database?",
      "points": 10,
      "difficulty": "easy",
      "options": [
        "A database that stores vectors",
        "A traditional SQL database",
        "A NoSQL database",
        "A graph database"
      ],
      "correctAnswer": 0,
      "explanation": "Vector databases are designed specifically to store and search high-dimensional vectors efficiently.",
      "hints": [
        "Think about what the name implies",
        "It's specialized for a specific type of data",
        "The data type is mentioned in the name"
      ]
    }
  ]
}
```

### Step 2: Use AssessmentSystem Component

```jsx
import AssessmentSystem from './components/tutorial/AssessmentSystem';

const handleComplete = (result) => {
  const { passed, score, timeSpent, gradedResults } = result;

  if (passed) {
    // Unlock next module
    console.log('Assessment passed with score:', score);
  }
};

<AssessmentSystem
  moduleId="module1"
  onComplete={handleComplete}
  minPassScore={70}
/>
```

### Question Types

**Multiple Choice**:
```json
{
  "type": "multiple-choice",
  "question": "...",
  "options": ["A", "B", "C", "D"],
  "correctAnswer": 0
}
```

**Multiple Answers**:
```json
{
  "type": "multiple-choice",
  "multipleAnswers": true,
  "question": "...",
  "options": ["A", "B", "C", "D"],
  "correctAnswers": [0, 2]
}
```

**True/False**:
```json
{
  "type": "true-false",
  "question": "...",
  "correctAnswer": true
}
```

**Code Challenge**:
```json
{
  "type": "code-challenge",
  "question": "...",
  "starterCode": "// Write your code here",
  "testCases": [
    { "input": "...", "expected": "..." }
  ]
}
```

---

## Achievement Integration

### Automatic Achievement Checking

The `TutorialWrapper` automatically checks for achievements after assessment completion. You can also manually check:

```jsx
import { checkAchievements } from './lib/achievementLogic';

const unlocked = checkAchievements({
  moduleId: 'module1',
  score: 95,
  timeSpent: 300000, // 5 minutes in ms
  passed: true
});

if (unlocked.length > 0) {
  // Show notifications
  unlocked.forEach(achievement => {
    console.log('Unlocked:', achievement.name);
  });
}
```

### Achievement Conditions

Achievements unlock automatically based on conditions:

- **Module Completion**: `{ type: 'module_complete', module: 'module1' }`
- **Perfect Score**: `{ type: 'perfect_score', module: 'module1' }`
- **Speed**: `{ type: 'speed_completion', maxTime: 300000 }`
- **First Attempt**: `{ type: 'first_attempt_pass' }`
- **All Modules**: `{ type: 'all_modules_complete', count: 6 }`
- **Readiness Level**: `{ type: 'readiness_level', level: 'expert' }`

### Displaying Achievement Notifications

```jsx
import AchievementNotification from './components/tutorial/AchievementNotification';

const [newAchievements, setNewAchievements] = useState([]);

{newAchievements.map(achievement => (
  <AchievementNotification
    key={achievement.id}
    achievement={achievement}
    onClose={() => removeAchievement(achievement.id)}
  />
))}
```

### Achievement System UI

```jsx
import AchievementSystem from './components/tutorial/AchievementSystem';

<AchievementSystem onClose={() => navigate('/tutorials')} />
```

---

## Contextual Help

### Using the Hook

```jsx
import { useContextualHelp } from './hooks/useContextualHelp';

const MyComponent = () => {
  const { openHelp, isHelpOpen, closeHelp } = useContextualHelp();

  return (
    <div>
      <button onClick={() => openHelp()}>Help</button>

      <HelpOverlay
        isOpen={isHelpOpen}
        onClose={closeHelp}
        initialContext="quiz-question" // Optional context ID
      />
    </div>
  );
};
```

### Keyboard Shortcuts

- **F1 or ?**: Open help overlay
- **ESC**: Close help overlay

### Help Tooltips

```jsx
import HelpTooltip, { HelpIcon, HelpLabel } from './components/tutorial/HelpTooltip';

// Simple icon with tooltip
<HelpIcon
  content="This is helpful information"
  title="Need Help?"
  position="top"
/>

// Label with help
<HelpLabel
  label="Vector Dimension"
  helpContent="The size of your embedding vectors"
  required={true}
/>

// Custom tooltip
<HelpTooltip
  content="Detailed help text here"
  title="Feature Name"
  position="right"
>
  <button>Hover me</button>
</HelpTooltip>
```

### Adding Help Content

Edit `src/data/helpContent.json`:

```json
{
  "help_topics": [
    {
      "id": "my-topic",
      "title": "My Topic",
      "category": "Getting Started",
      "content": "Detailed explanation...",
      "keywords": ["search", "terms"],
      "relatedTopics": ["other-topic-id"]
    }
  ],
  "contextual_help": {
    "my-context": {
      "title": "Context Title",
      "content": "Context-specific help",
      "icon": "💡"
    }
  }
}
```

---

## Readiness Assessment

### Triggering Readiness Assessment

```jsx
import ReadinessAssessment from './components/tutorial/ReadinessAssessment';

const [showReadiness, setShowReadiness] = useState(false);

// Check if all modules are complete
const allModulesComplete = ['module1', 'module2', 'module3', 'module4', 'module5', 'module6']
  .every(id => assessmentState.hasPassedModule(id));

if (allModulesComplete) {
  setShowReadiness(true);
}

<ReadinessAssessment
  onClose={() => setShowReadiness(false)}
  onContinue={(evaluation) => {
    console.log('Readiness evaluation:', evaluation);
    // Navigate to certificate or next section
  }}
  userInfo={{
    name: 'John Doe',
    email: 'john@example.com'
  }}
/>
```

### Evaluation Components

The readiness assessment includes multiple views:

1. **Overview**: Overall score and skill area breakdown
2. **Skills Checklist**: 14-item production readiness checklist
3. **Detailed Report**: Module scores, strengths, and gaps
4. **Recommendations**: Personalized guidance
5. **Certificate**: Downloadable certificate (if score ≥ 60%)

### Understanding Results

```jsx
const handleReadinessComplete = (evaluation) => {
  console.log('Overall Score:', evaluation.overallScore);
  console.log('Proficiency:', evaluation.proficiencyLevel.label);
  console.log('Ready for Production:', evaluation.readyForProduction);
  console.log('Recommended:', evaluation.recommendedForProduction);
  console.log('Skill Gaps:', evaluation.skillGaps);
};
```

---

## Complete Example

Here's a complete integration example:

```jsx
import React, { useState } from 'react';
import TutorialWrapper from './components/tutorial/TutorialWrapper';
import { HelpIcon } from './components/tutorial/HelpTooltip';

const Module1GettingStarted = ({ onComplete }) => {
  const [currentStep, setCurrentStep] = useState(0);

  const steps = [
    {
      title: 'Introduction',
      content: 'Learn about vector databases...'
    },
    {
      title: 'Creating a Database',
      content: 'Create your first database...'
    },
    // ... more steps
  ];

  return (
    <TutorialWrapper
      moduleId="module1"
      moduleName="Module 1: Getting Started"
      onComplete={onComplete}
    >
      {/* Module Content */}
      <div className="space-y-6">
        {/* Progress */}
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-bold">
              {steps[currentStep].title}
            </h2>
            <HelpIcon
              content="Click through each step to learn the concepts"
              title="How to use this module"
            />
          </div>

          <div className="prose max-w-none">
            {steps[currentStep].content}
          </div>
        </div>

        {/* Navigation */}
        <div className="flex gap-4 justify-between">
          <button
            onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
            disabled={currentStep === 0}
            className="px-6 py-3 bg-gray-600 text-white rounded-lg disabled:opacity-50"
          >
            Previous
          </button>
          <button
            onClick={() => setCurrentStep(Math.min(steps.length - 1, currentStep + 1))}
            disabled={currentStep === steps.length - 1}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg disabled:opacity-50"
          >
            Next
          </button>
        </div>
      </div>
    </TutorialWrapper>
  );
};

export default Module1GettingStarted;
```

---

## Best Practices

### 1. Assessment Placement
- Place assessment trigger after all module content
- Allow users to review content before assessment
- Enable retakes for improvement

### 2. Achievement Design
- Make achievements discoverable (show locked badges)
- Provide immediate feedback on unlock
- Balance difficulty (some easy, some challenging)

### 3. Help System
- Add help icons near complex features
- Keep help content concise and actionable
- Organize by category for easy navigation

### 4. Progress Tracking
- Show clear progress indicators
- Save progress automatically
- Display completion status prominently

### 5. User Experience
- Provide keyboard shortcuts (F1, ESC, Tab)
- Use animations for engagement (but don't overdo it)
- Make retrying assessments easy
- Celebrate successes with achievements

---

## Troubleshooting

### Assessment not loading
- Check that quiz JSON file exists in `src/data/quizzes/`
- Verify moduleId matches between component and JSON
- Check browser console for errors

### Achievements not unlocking
- Verify achievement conditions in `achievements.json`
- Check that context object has required fields
- Use `checkAchievements()` with console.log for debugging

### Help system not working
- Ensure `useContextualHelp` hook is called in component
- Check that `HelpOverlay` is rendered
- Verify F1 key isn't intercepted by browser

### Readiness assessment errors
- Ensure at least one module assessment is completed
- Check localStorage for assessment data
- Verify all module IDs are consistent

---

## API Reference

### TutorialWrapper
```typescript
<TutorialWrapper
  moduleId: string
  moduleName: string
  children: ReactNode
  onComplete?: () => void
  showReadinessWhenComplete?: boolean
/>
```

### AssessmentSystem
```typescript
<AssessmentSystem
  moduleId: string
  onComplete: (result: AssessmentResult) => void
  onRetry?: () => void
  minPassScore?: number
/>
```

### AchievementSystem
```typescript
<AchievementSystem
  onClose?: () => void
/>
```

### ReadinessAssessment
```typescript
<ReadinessAssessment
  onClose?: () => void
  onContinue?: (evaluation: Evaluation) => void
  userInfo?: { name?: string; email?: string }
/>
```

### useContextualHelp
```typescript
const {
  isHelpOpen: boolean,
  openHelp: (context?: string) => void,
  closeHelp: () => void,
  searchHelp: (query: string) => Topic[],
  getHelpTopic: (id: string) => Topic,
  getRandomTip: () => string
} = useContextualHelp();
```

---

## Need More Help?

- Check individual component documentation in their respective files
- Review example integrations in the codebase
- Press F1 in the tutorial for contextual help
- See `../../docs/SPRINT_5_FINAL_PLAN.md` for implementation details
