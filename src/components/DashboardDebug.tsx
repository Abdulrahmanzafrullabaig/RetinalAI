import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';

const DashboardDebug: React.FC = () => {
  const { user } = useAuth();
  const [debugInfo, setDebugInfo] = useState<any>({});
  const [apiTest, setApiTest] = useState<string>('Not tested');

  useEffect(() => {
    const testConnection = async () => {
      try {
        // Test basic connection
        const response = await fetch('/api/user', {
          method: 'GET',
          credentials: 'include',
        });
        
        const data = await response.json();
        setApiTest(`Status: ${response.status}, Data: ${JSON.stringify(data)}`);
      } catch (err) {
        setApiTest(`Error: ${err}`);
      }
    };

    setDebugInfo({
      user: user,
      userExists: !!user,
      userRole: user?.role,
      timestamp: new Date().toISOString()
    });

    testConnection();
  }, [user]);

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-2xl font-bold mb-6">Dashboard Debug Information</h1>
        
        <div className="bg-white p-6 rounded-lg shadow mb-6">
          <h2 className="text-lg font-semibold mb-4">Auth Context</h2>
          <pre className="bg-gray-100 p-4 rounded text-sm overflow-auto">
            {JSON.stringify(debugInfo, null, 2)}
          </pre>
        </div>

        <div className="bg-white p-6 rounded-lg shadow mb-6">
          <h2 className="text-lg font-semibold mb-4">API Test</h2>
          <p className="text-sm">{apiTest}</p>
        </div>

        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-lg font-semibold mb-4">Environment</h2>
          <ul className="text-sm space-y-2">
            <li><strong>Current URL:</strong> {window.location.href}</li>
            <li><strong>Base URL:</strong> {window.location.origin}</li>
            <li><strong>User Agent:</strong> {navigator.userAgent}</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default DashboardDebug;