import React, { useState } from 'react';
import { useTutorialState } from '../hooks/useTutorialState';

const PreferencesPanel = () => {
  const { state, updatePreference } = useTutorialState();
  const [localPreferences, setLocalPreferences] = useState(state.preferences);
  
  const handlePreferenceChange = (key, value) => {
    setLocalPreferences(prev => ({ ...prev, [key]: value }));
    updatePreference(key, value);
  };
  
  const handleSavePreferences = () => {
    // In a real implementation, we would save to the backend or localStorage
    Object.entries(localPreferences).forEach(([key, value]) => {
      updatePreference(key, value);
    });
    
    // Show a confirmation
    const saveButton = document.getElementById('save-preferences-btn');
    if (saveButton) {
      const originalText = saveButton.textContent;
      saveButton.textContent = 'Saved!';
      setTimeout(() => {
        saveButton.textContent = originalText;
      }, 2000);
    }
  };

  return (
    <div className="module-card">
      <h2 className="text-xl font-semibold text-gray-800 mb-6">Tutorial Preferences</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Left Column - Learning Preferences */}
        <div>
          <h3 className="font-medium text-gray-800 mb-4 pb-2 border-b border-gray-200">Learning Preferences</h3>
          
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Experience Level
              </label>
              <div className="grid grid-cols-3 gap-2">
                {[
                  { value: 'beginner', label: 'Beginner', description: 'New to vector databases' },
                  { value: 'intermediate', label: 'Intermediate', description: 'Some experience' },
                  { value: 'advanced', label: 'Advanced', description: 'Expert level' }
                ].map((level) => (
                  <button
                    key={level.value}
                    onClick={() => handlePreferenceChange('experienceLevel', level.value)}
                    className={`p-3 rounded-lg border text-left transition-all ${
                      localPreferences.experienceLevel === level.value
                        ? 'border-blue-500 bg-blue-50 ring-2 ring-blue-200'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="font-medium text-gray-800 text-sm">{level.label}</div>
                    <div className="text-xs text-gray-600 mt-1">{level.description}</div>
                  </button>
                ))}
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Preferred Language
              </label>
              <div className="grid grid-cols-4 gap-2">
                {[
                  { value: 'javascript', label: 'JavaScript', icon: 'JS' },
                  { value: 'python', label: 'Python', icon: 'PY' },
                  { value: 'go', label: 'Go', icon: 'GO' },
                  { value: 'java', label: 'Java', icon: 'JA' }
                ].map((lang) => (
                  <button
                    key={lang.value}
                    onClick={() => handlePreferenceChange('preferredLanguage', lang.value)}
                    className={`p-3 rounded-lg border text-center transition-all ${
                      localPreferences.preferredLanguage === lang.value
                        ? 'border-blue-500 bg-blue-50 ring-2 ring-blue-200'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="font-medium text-gray-800 text-xs">{lang.icon}</div>
                    <div className="text-xs text-gray-600 mt-1">{lang.label}</div>
                  </button>
                ))}
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Use Case Focus
              </label>
              <select
                value={localPreferences.useCaseFocus}
                onChange={(e) => handlePreferenceChange('useCaseFocus', e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-md"
              >
                <option value="general">General Purpose</option>
                <option value="retrieval">Information Retrieval</option>
                <option value="recommendation">Recommendations</option>
                <option value="semantic">Semantic Search</option>
                <option value="multimodal">Multimodal Applications</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Learning Pace
              </label>
              <div className="grid grid-cols-3 gap-2">
                {[
                  { value: 'slow', label: 'Slow' },
                  { value: 'normal', label: 'Normal' },
                  { value: 'fast', label: 'Fast' }
                ].map((pace) => (
                  <button
                    key={pace.value}
                    onClick={() => handlePreferenceChange('pace', pace.value)}
                    className={`p-2 rounded-lg border text-center transition-all ${
                      localPreferences.pace === pace.value
                        ? 'border-blue-500 bg-blue-50 ring-2 ring-blue-200'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="font-medium text-gray-800 text-sm">{pace.label}</div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
        
        {/* Right Column - Interface Preferences */}
        <div>
          <h3 className="font-medium text-gray-800 mb-4 pb-2 border-b border-gray-200">Interface Preferences</h3>
          
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Theme
              </label>
              <div className="grid grid-cols-3 gap-2">
                {[
                  { value: 'light', label: 'Light' },
                  { value: 'dark', label: 'Dark' },
                  { value: 'auto', label: 'Auto' }
                ].map((theme) => (
                  <button
                    key={theme.value}
                    onClick={() => handlePreferenceChange('theme', theme.value)}
                    className={`p-2 rounded-lg border text-center transition-all ${
                      localPreferences.theme === theme.value
                        ? 'border-blue-500 bg-blue-50 ring-2 ring-blue-200'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="font-medium text-gray-800 text-sm">{theme.label}</div>
                  </button>
                ))}
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Font Size
              </label>
              <div className="flex items-center space-x-4">
                <input
                  type="range"
                  min="12"
                  max="20"
                  value={localPreferences.fontSize}
                  onChange={(e) => handlePreferenceChange('fontSize', parseInt(e.target.value))}
                  className="flex-1"
                />
                <span className="text-sm text-gray-600 w-12">{localPreferences.fontSize}px</span>
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Notifications
              </label>
              <div className="flex items-center">
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={localPreferences.notifications}
                    onChange={(e) => handlePreferenceChange('notifications', e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
                  <span className="ml-3 text-sm font-medium text-gray-700">
                    {localPreferences.notifications ? 'Enabled' : 'Disabled'}
                  </span>
                </label>
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Auto-save Progress
              </label>
              <div className="flex items-center">
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={localPreferences.autoSave}
                    onChange={(e) => handlePreferenceChange('autoSave', e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
                  <span className="ml-3 text-sm font-medium text-gray-700">
                    {localPreferences.autoSave ? 'Enabled' : 'Disabled'}
                  </span>
                </label>
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Code Editor Theme
              </label>
              <select
                value={localPreferences.editorTheme || 'jade-light'}
                onChange={(e) => handlePreferenceChange('editorTheme', e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-md"
              >
                <option value="jade-light">Jade Light</option>
                <option value="jade-dark">Jade Dark</option>
                <option value="vs-light">VS Light</option>
                <option value="vs-dark">VS Dark</option>
                <option value="hc-black">High Contrast</option>
              </select>
            </div>
          </div>
        </div>
      </div>
      
      <div className="mt-8 pt-6 border-t border-gray-200 flex justify-end">
        <button
          id="save-preferences-btn"
          onClick={handleSavePreferences}
          className="btn-primary px-6 py-2"
        >
          Save Preferences
        </button>
      </div>
    </div>
  );
};

export default PreferencesPanel;