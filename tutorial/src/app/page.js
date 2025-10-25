'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { TutorialProvider } from '../contexts/TutorialContext';
import TutorialHeader from '../components/TutorialHeader';
import InstructionsPanel from '../components/InstructionsPanel';
import VisualDashboard from '../components/VisualDashboard';
import CodeEditor from '../components/CodeEditor';
import LivePreviewPanel from '../components/LivePreviewPanel';
import TutorialState from '../lib/tutorialState';

export default function Home() {
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

  const handleResetTutorial = () => {
    // Reset to initial state
    const initialState = TutorialState.getInitialState();
    setTutorialState(initialState);
    setCurrentModule(0);
    setCurrentStep(0);
    
    // Clear any saved state in localStorage
    TutorialState.clearStorage();
  };

  return (
    <TutorialProvider value={{ tutorialState, setTutorialState, currentModule, setCurrentModule, currentStep, setCurrentStep }}>
      <div className="tutorial-container">
        <TutorialHeader onResetTutorial={handleResetTutorial} />
        
        <main className="tutorial-main">
          <InstructionsPanel />
          
          <div className="tutorial-content">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="space-y-6"
            >
              <VisualDashboard />
              <CodeEditor />
              <LivePreviewPanel />
            </motion.div>
          </div>
        </main>
      </div>
    </TutorialProvider>
  );
}