'use client';

import React, { useState, useEffect } from 'react';
import { TutorialProvider } from '../../contexts/TutorialContext';
import TutorialHeader from '../../components/TutorialHeader';
import InstructionsPanel from '../../components/InstructionsPanel';
import VisualDashboard from '../../components/VisualDashboard';
import CodeEditor from '../../components/CodeEditor';
import LivePreviewPanel from '../../components/LivePreviewPanel';
import TutorialState from '../../lib/tutorialState';

const TutorialPage = () => {
  const [tutorialState, setTutorialState] = useState(TutorialState.getInitialState());
  const [currentModule, setCurrentModule] = useState(0);
  const [currentStep, setCurrentStep] = useState(0);

  // Load initial state from localStorage or set defaults
  useEffect(() => {
    const savedState = TutorialState.loadFromStorage();
    if (savedState) {
      setTutorialState(savedState);
    }
  }, []);

  // Save state whenever it changes
  useEffect(() => {
    TutorialState.saveToStorage(tutorialState);
  }, [tutorialState]);

  return (
    <TutorialProvider value={{ tutorialState, setTutorialState, currentModule, setCurrentModule, currentStep, setCurrentStep }}>
      <div className="tutorial-container">
        <TutorialHeader />
        
        <main className="tutorial-main">
          <InstructionsPanel />
          
          <div className="tutorial-content">
            <div className="space-y-6">
              <VisualDashboard />
              <CodeEditor />
              <LivePreviewPanel />
            </div>
          </div>
        </main>
      </div>
    </TutorialProvider>
  );
};

export default TutorialPage;