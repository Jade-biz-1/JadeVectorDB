import React from 'react';
import Link from 'next/link';

const TutorialHeader = ({ onResetTutorial }) => {
  const handleResetClick = () => {
    if (onResetTutorial) {
      onResetTutorial();
    } else {
      // Default reset behavior: reload the page to reset tutorial state
      if (typeof window !== 'undefined') {
        window.location.reload();
      }
    }
  };

  const handleModulesClick = (e) => {
    e.preventDefault();
    // Scroll to the modules section or open modules sidebar
    const modulesSection = document.querySelector('.tutorial-sidebar');
    if (modulesSection) {
      modulesSection.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const handleResourcesClick = (e) => {
    e.preventDefault();
    // In the full tutorial, this could navigate to resources section
    alert('Resources section would open here');
  };

  const handleDocumentationClick = (e) => {
    e.preventDefault();
    // In the full tutorial, this could open documentation
    alert('Documentation section would open here');
  };

  return (
    <header className="tutorial-header">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className="bg-blue-600 text-white p-2 rounded-lg font-bold text-lg">
            JDB
          </div>
          <h1 className="text-xl font-bold text-gray-800">JadeVectorDB Interactive Tutorial</h1>
        </div>
        
        <nav className="hidden md:flex items-center space-x-6">
          <a href="#" onClick={handleModulesClick} className="text-gray-600 hover:text-gray-900 transition-colors">Modules</a>
          <a href="#" onClick={handleDocumentationClick} className="text-gray-600 hover:text-gray-900 transition-colors">Documentation</a>
          <a href="#" onClick={handleResourcesClick} className="text-gray-600 hover:text-gray-900 transition-colors">Resources</a>
          <button onClick={handleResetClick} className="btn-secondary">Reset Tutorial</button>
        </nav>
      </div>
    </header>
  );
};

export default TutorialHeader;