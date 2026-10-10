import { useState } from 'react';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import FileUpload from './components/FileUpload';
import Dashboard from './components/Dashboard';
import Analytics from './components/Analytics';
import ChatPanel from './components/ChatPanel';
import InsightsPanel from './components/InsightsPanel';
import type { UploadResponse } from './types';
import { getPreview } from './api';

export default function App() {
  const [currentData, setCurrentData] = useState<UploadResponse | null>(null);
  const [activeTab, setActiveTab] = useState<string>('upload');

  const handleDataLoaded = (data: UploadResponse) => {
    setCurrentData(data);
    setActiveTab('dashboard');
  };

  const handleDatasetChanged = async (datasetName: string) => {
    try {
      const preview = await getPreview(datasetName);
      if (currentData) {
        setCurrentData({
          ...currentData,
          filename: datasetName,
          row_count: preview.length,
          preview,
        });
      }
    } catch {
      // Reload via standard preview if possible
    }
  };

  return (
    <div className="min-h-screen bg-[#F6F7F9] flex flex-col font-sans text-[#172033]">
      <Navbar 
        currentFile={currentData?.filename || null} 
        onNavigate={(tab) => setActiveTab(tab)} 
        onDatasetChanged={handleDatasetChanged}
      />
      
      <div className="flex flex-1">
        <Sidebar
          activeTab={activeTab}
          onTabChange={setActiveTab}
          hasData={!!currentData}
        />

        <main className="flex-1 p-5 md:p-7 overflow-y-auto max-w-7xl mx-auto w-full">
          {activeTab === 'upload' && (
            <FileUpload
              onDataLoaded={handleDataLoaded}
              currentData={currentData}
            />
          )}

          {activeTab === 'dashboard' && currentData && (
            <Dashboard
              data={currentData}
              onNavigate={(tab) => setActiveTab(tab)}
            />
          )}

          {/* WORKSPACE: AI Chat */}
          {activeTab === 'chat' && currentData && (
            <ChatPanel data={currentData} />
          )}

          {/* WORKSPACE: Analytics */}
          {activeTab === 'analytics' && currentData && (
            <Analytics data={currentData} />
          )}

          {/* INTELLIGENCE: Insights */}
          {activeTab === 'insights' && currentData && (
            <InsightsPanel data={currentData} />
          )}

          {!currentData && activeTab !== 'upload' && (
            <div className="text-center py-20 bg-white rounded-xl border border-slate-200">
              <p className="text-slate-600 mb-4">No dataset loaded yet.</p>
              <button
                onClick={() => setActiveTab('upload')}
                className="px-5 py-2.5 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 text-sm font-medium transition"
              >
                Go to Upload & Sample Data
              </button>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
