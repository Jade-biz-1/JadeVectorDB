'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';

const WelcomePage = () => {
  const router = useRouter();
  const [experienceLevel, setExperienceLevel] = useState('beginner');
  const [preferredLanguage, setPreferredLanguage] = useState('javascript');
  const [useCaseFocus, setUseCaseFocus] = useState('general');

  const handleStartTutorial = () => {
    // In a real implementation, we would save these preferences
    // and redirect to the main tutorial
    router.push('/');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <div className="max-w-4xl w-full bg-white rounded-2xl shadow-xl overflow-hidden">
        <div className="md:flex">
          <div className="md:w-2/5 bg-gradient-to-br from-blue-600 to-indigo-700 text-white p-8 flex flex-col justify-center">
            <div className="text-center">
              <div className="bg-white bg-opacity-20 rounded-full w-24 h-24 flex items-center justify-center mx-auto mb-6">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                </svg>
              </div>
              <h1 className="text-3xl font-bold mb-2">JadeVectorDB</h1>
              <h2 className="text-xl font-semibold mb-4">Interactive Tutorial</h2>
              <p className="opacity-90">
                Learn vector databases through hands-on experience with guided tutorials and real-time visualizations.
              </p>
            </div>
          </div>
          
          <div className="md:w-3/5 p-8">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-2">Welcome to the Tutorial</h2>
              <p className="text-gray-600">
                Let's customize your learning experience
              </p>
            </div>
            
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-medium text-gray-800 mb-3">Experience Level</h3>
                <div className="grid grid-cols-3 gap-3">
                  {[
                    { id: 'beginner', label: 'Beginner', description: 'New to vector databases' },
                    { id: 'intermediate', label: 'Intermediate', description: 'Some experience' },
                    { id: 'advanced', label: 'Advanced', description: 'Expert level' }
                  ].map((level) => (
                    <button
                      key={level.id}
                      className={`p-4 rounded-lg border text-left transition-all ${
                        experienceLevel === level.id
                          ? 'border-blue-500 bg-blue-50 ring-2 ring-blue-200'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                      onClick={() => setExperienceLevel(level.id)}
                    >
                      <div className="font-medium text-gray-800">{level.label}</div>
                      <div className="text-sm text-gray-600">{level.description}</div>
                    </button>
                  ))}
                </div>
              </div>
              
              <div>
                <h3 className="text-lg font-medium text-gray-800 mb-3">Preferred Language</h3>
                <div className="grid grid-cols-4 gap-3">
                  {[
                    { id: 'javascript', label: 'JavaScript' },
                    { id: 'python', label: 'Python' },
                    { id: 'go', label: 'Go' },
                    { id: 'java', label: 'Java' }
                  ].map((lang) => (
                    <button
                      key={lang.id}
                      className={`p-3 rounded-lg border text-center transition-all ${
                        preferredLanguage === lang.id
                          ? 'border-blue-500 bg-blue-50 ring-2 ring-blue-200'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                      onClick={() => setPreferredLanguage(lang.id)}
                    >
                      <div className="font-medium text-gray-800 text-sm">{lang.label}</div>
                    </button>
                  ))}
                </div>
              </div>
              
              <div>
                <h3 className="text-lg font-medium text-gray-800 mb-3">Use Case Focus</h3>
                <div className="grid grid-cols-2 gap-3">
                  {[
                    { id: 'general', label: 'General Purpose' },
                    { id: 'retrieval', label: 'Information Retrieval' },
                    { id: 'recommendation', label: 'Recommendations' },
                    { id: 'semantic', label: 'Semantic Search' }
                  ].map((focus) => (
                    <button
                      key={focus.id}
                      className={`p-3 rounded-lg border text-center transition-all ${
                        useCaseFocus === focus.id
                          ? 'border-blue-500 bg-blue-50 ring-2 ring-blue-200'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                      onClick={() => setUseCaseFocus(focus.id)}
                    >
                      <div className="font-medium text-gray-800 text-sm">{focus.label}</div>
                    </button>
                  ))}
                </div>
              </div>
            </div>
            
            <div className="mt-8 flex flex-col sm:flex-row gap-3">
              <button
                onClick={handleStartTutorial}
                className="flex-1 btn-primary py-3 px-6 text-center"
              >
                Start Learning
              </button>
              <Link
                href="/"
                className="flex-1 btn-secondary py-3 px-6 text-center"
              >
                Skip Setup
              </Link>
            </div>
            
            <div className="mt-6 text-center text-sm text-gray-500">
              <p>You can change these settings anytime in the tutorial preferences.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WelcomePage;