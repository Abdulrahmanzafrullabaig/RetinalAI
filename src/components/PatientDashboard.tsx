import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Upload, History, Share, Calendar, Eye, RefreshCcw } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import Header from './Header';
import { API_URL } from '../config';

interface AnalysisReport {
  id: number;
  date: string;
  result: string;
  confidence: string;
  status: string; // From backend, could be 'completed'
  doctor_notes?: string;
}

const PatientDashboard = () => {
  const { user } = useAuth();
  const [recentAnalyses, setRecentAnalyses] = useState<AnalysisReport[]>([]);
  const [allAnalyses, setAllAnalyses] = useState<AnalysisReport[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [authChecked, setAuthChecked] = useState(false);

  const fetchPatientData = async () => {
    setLoading(true);
    setError('');

    console.log('Fetching patient data...');

    try {
      const response = await fetch(`${API_URL}/api/results`, {
        method: 'GET',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      console.log('Response status:', response.status);

      if (response.ok) {
        const data: AnalysisReport[] = await response.json();
        console.log('Received data:', data);
        setAllAnalyses(data); // Store all analyses
        setRecentAnalyses(data.slice(0, 3)); // Get up to 3 most recent analyses
      } else if (response.status === 401) {
        setError('Session expired. Please log in again.');
        console.error('Authentication required');
      } else if (response.status === 403) {
        setError('Access denied. Please check your permissions.');
      } else if (response.status >= 500) {
        setError('Server error. Please try again later.');
      } else {
        try {
          const errorData = await response.json();
          setError(errorData.message || `Server error: ${response.status}`);
        } catch {
          setError(`Server error: ${response.status} - ${response.statusText}`);
        }
      }
    } catch (err) {
      console.error('Error fetching patient data:', err);
      if (err instanceof TypeError && err.message.includes('fetch')) {
        setError('Cannot connect to server. Please check if the backend is running on port 5000.');
      } else {
        setError('Network error. Please check your internet connection.');
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Wait for auth context to determine if user is logged in
    const checkAuthAndFetch = async () => {
      // Give auth context time to load
      await new Promise(resolve => setTimeout(resolve, 200));

      setAuthChecked(true);

      if (user) {
        console.log('User authenticated:', user);
        fetchPatientData();
      } else {
        console.log('User not authenticated');
        setLoading(false);
        setError('Please log in to view your dashboard.');
      }
    };

    // Initial fetch
    checkAuthAndFetch();

    // Poll for updates every 30 seconds
    const intervalId = setInterval(() => {
      if (user) {
        // Fetch quietly (without setting full loading state to avoid UI flicker)
        console.log('Polling patient data...');
        fetch(`${API_URL}/api/results`, {
          method: 'GET',
          credentials: 'include',
          headers: {
            'Content-Type': 'application/json',
          },
        })
          .then(response => {
            if (response.ok) return response.json();
            throw new Error('Network response was not ok');
          })
          .then(data => {
            setAllAnalyses(data as AnalysisReport[]); // Store all analyses
            setRecentAnalyses((data as AnalysisReport[]).slice(0, 3));
          })
          .catch(err => console.error('Polling error:', err));
      }
    }, 30000);

    return () => clearInterval(intervalId);
  }, [user]);

  const getResultColor = (result: string) => {
    // Handle clinical prediction results
    if (result === 'Positive for Diabetic Retinopathy') {
      return 'text-red-600 bg-red-50 border-red-200';
    }
    if (result === 'Negative for Diabetic Retinopathy') {
      return 'text-green-600 bg-green-50 border-green-200';
    }

    // Handle fundus image analysis results
    if (result === 'None' || result === 'Not a fundus image') {
      return 'text-gray-600 bg-gray-50 border-gray-200';
    }

    const shortResult = result.split(' - ')[0];
    switch (shortResult) {
      case 'No DR': return 'text-green-600 bg-green-50 border-green-200';
      case 'Mild DR': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'Moderate DR': return 'text-orange-600 bg-orange-50 border-orange-200';
      case 'Severe DR': return 'text-red-600 bg-red-50 border-red-200';
      case 'Proliferative DR': return 'text-purple-600 bg-purple-50 border-purple-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const totalAnalyses = allAnalyses.length;
  const lastAnalysisDate = allAnalyses.length > 0 ? allAnalyses[0].date : 'N/A';
  // Determine monitoring status - this is a simplified example
  const monitoringStatus = totalAnalyses > 0 && (allAnalyses[0].result === 'No DR - No visible abnormalities' || allAnalyses[0].result === 'Negative for Diabetic Retinopathy') ? 'Monitoring' : 'Needs Review';
  const monitoringStatusColor = monitoringStatus === 'Monitoring' ? 'bg-green-100 text-green-700' : 'bg-amber-100 text-amber-700';

  // Fix date calculation to show whole days and handle today's date
  const lastAnalysisText = lastAnalysisDate !== 'N/A' ? (
    Math.floor((new Date().getTime() - new Date(lastAnalysisDate).getTime()) / (1000 * 3600 * 24)) === 0 ?
    'Today' :
    `${Math.floor((new Date().getTime() - new Date(lastAnalysisDate).getTime()) / (1000 * 3600 * 24))} days ago`
  ) : 'N/A';


  // Show loading while auth is being checked or data is being fetched
  if (!authChecked || loading) {
    return (
      <div className="min-h-screen bg-background-100 flex items-center justify-center">
        <div className="flex flex-col items-center space-y-4 text-primary-700">
          <RefreshCcw className="h-8 w-8 animate-spin" />
          <span className="text-lg font-medium">
            {!authChecked ? 'Checking authentication...' : 'Loading Dashboard...'}
          </span>
          <span className="text-sm text-primary-600">
            {!authChecked ? 'Please wait' : 'Fetching your health data'}
          </span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-background-100 flex items-center justify-center">
        <div className="bg-red-100 border border-red-400 text-red-700 px-6 py-4 rounded-lg max-w-md text-center" role="alert">
          <strong className="font-bold block mb-2">Dashboard Error</strong>
          <span className="block mb-4">{error}</span>
          <div className="flex gap-2 justify-center">
            <button
              onClick={fetchPatientData}
              className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded transition-colors"
            >
              Try Again
            </button>
            {error.includes('log in') && (
              <button
                onClick={() => window.location.href = '/login'}
                className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded transition-colors"
              >
                Login
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-white">
      <Header />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        {/* Welcome Section */}
        <div className="mb-10 animate-fade-in-up">
          <h1 className="text-4xl font-bold text-gray-900 mb-3">
            Welcome back, <span className="text-secondary-500">{user?.username}</span>
          </h1>
          <p className="text-lg text-gray-500">
            Track your retinal health with AI-powered analysis and monitoring.
          </p>
        </div>

        {/* Quick Actions */}
        <div className="grid lg:grid-cols-4 gap-6 mb-10">
          {[
            { to: '/new-analysis', icon: Upload, title: 'New Analysis', desc: 'Choose image or clinical data analysis', color: 'bg-secondary-500', delay: '0ms' },
            { to: '/results', icon: History, title: 'Analysis History', desc: 'View all your previous results', color: 'bg-primary-500', delay: '100ms' },
            { to: '/shared-reports', icon: Share, title: 'Shared Reports', desc: 'View reports shared with doctor', color: 'bg-primary-600', delay: '200ms' },
            { to: '/appointments', icon: Calendar, title: 'Appointments', desc: 'Schedule follow-up appointments', color: 'bg-primary-700', delay: '300ms' },
          ].map((action, index) => {
            const ActionWrapper = action.to ? Link : 'div';
            const props = action.to ? { to: action.to } : {};
            return (
              <ActionWrapper
                key={index}
                {...props}
                className={`bg-white rounded-2xl p-6 shadow-card border-2 border-gray-100 hover:shadow-card-hover transition-all duration-300 transform hover:scale-105 group animate-fade-in-up`}
                style={{ animationDelay: action.delay }}
              >
                <div className="flex items-center justify-between mb-4">
                  <div className={`${action.color} rounded-xl p-4 group-hover:scale-110 group-hover:rotate-6 transition-all duration-300 shadow-lg`}>
                    <action.icon className="h-7 w-7 text-white" />
                  </div>
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-2">{action.title}</h3>
                <p className="text-sm text-gray-500">{action.desc}</p>
              </ActionWrapper>
            );
          })}
        </div>

        {/* Recent Analyses */}
        <div className="grid lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2 animate-fade-in-up" style={{ animationDelay: '400ms' }}>
            <div className="bg-white rounded-2xl shadow-card border-2 border-gray-100 p-8">
              <div className="flex items-center justify-between mb-8">
                <h2 className="text-2xl font-bold text-gray-900">Recent Analyses</h2>
                <button
                  onClick={fetchPatientData}
                  disabled={loading}
                  className="text-gray-500 hover:text-primary-600 transition-all duration-300 transform hover:scale-110 disabled:opacity-50 p-2 rounded-lg hover:bg-gray-50"
                  title="Refresh data"
                >
                  <RefreshCcw className={`h-5 w-5 ${loading ? 'animate-spin' : ''}`} />
                </button>
              </div>
              <div className="space-y-4">
                {recentAnalyses.length > 0 ? (
                  recentAnalyses.map((analysis, index) => (
                    <div
                      key={analysis.id}
                      className="flex items-center justify-between p-5 bg-gray-50 rounded-xl border-2 border-gray-100 hover:border-primary-300 hover:shadow-md transition-all duration-300 transform hover:scale-[1.02] animate-fade-in-up"
                      style={{ animationDelay: `${500 + index * 100}ms` }}
                    >
                      <div className="flex items-center space-x-4">
                        <div className="bg-primary-500 rounded-xl p-3 shadow-md">
                          <Eye className="h-6 w-6 text-white" />
                        </div>
                        <div>
                          <p className="font-bold text-gray-900">{analysis.date}</p>
                          <p className="text-sm text-gray-500">
                            {analysis.confidence !== 'N/A' ? `Confidence: ${analysis.confidence}` : 'Clinical Analysis'}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-3">
                        <span className={`px-4 py-2 rounded-xl text-xs font-bold border-2 ${getResultColor(analysis.result)}`}>
                          {analysis.result.length > 30 ?
                            `${analysis.result.substring(0, 30)}...` :
                            analysis.result
                          }
                        </span>
                        <Link
                          to={`/result/${analysis.id}`}
                          className="text-primary-600 hover:text-primary-700 text-sm font-bold whitespace-nowrap transition-colors duration-300"
                        >
                          View Details →
                        </Link>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-16 animate-fade-in-up">
                    <div className="bg-primary-100 rounded-full p-6 w-20 h-20 mx-auto mb-6 flex items-center justify-center">
                      <Eye className="h-10 w-10 text-primary-500" />
                    </div>
                    <h3 className="text-xl font-bold text-gray-900 mb-3">No analyses yet</h3>
                    <p className="text-gray-500 mb-6">Start your first retinal health analysis</p>
                    <Link
                      to="/new-analysis"
                      className="inline-flex items-center px-6 py-3 bg-secondary-500 hover:bg-secondary-600 text-white rounded-xl font-bold transition-all duration-300 transform hover:scale-105 shadow-lg"
                    >
                      <Upload className="h-5 w-5 mr-2" />
                      Start Analysis
                    </Link>
                  </div>
                )}
              </div>
              {allAnalyses.length > 3 && (
                <div className="mt-8 text-center">
                  <Link
                    to="/results"
                    className="text-primary-600 hover:text-primary-700 font-bold transition-colors duration-300 inline-flex items-center space-x-2 group"
                  >
                    <span>View All Results</span>
                    <span className="group-hover:translate-x-1 transition-transform">→</span>
                  </Link>
                </div>
              )}
            </div>
          </div>

          {/* Health Summary */}
          <div className="space-y-6 animate-fade-in-up" style={{ animationDelay: '500ms' }}>
            <div className="bg-white rounded-2xl shadow-card border-2 border-gray-100 p-6">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-bold text-gray-900">Health Summary</h3>
                <button
                  onClick={fetchPatientData}
                  disabled={loading}
                  className="text-gray-500 hover:text-primary-600 transition-all duration-300 transform hover:scale-110 disabled:opacity-50 p-2 rounded-lg hover:bg-gray-50"
                  title="Refresh data"
                >
                  <RefreshCcw className={`h-5 w-5 ${loading ? 'animate-spin' : ''}`} />
                </button>
              </div>
              <div className="space-y-5">
                <div className="flex justify-between items-center pb-4 border-b-2 border-gray-100">
                  <span className="text-gray-500 font-medium">Total Analyses</span>
                  <span className="font-bold text-2xl text-secondary-500">{totalAnalyses}</span>
                </div>
                <div className="flex justify-between items-center pb-4 border-b-2 border-gray-100">
                  <span className="text-gray-500 font-medium">Last Analysis</span>
                  <span className="font-bold text-gray-900 text-right text-sm">
                    {lastAnalysisDate !== 'N/A' ? lastAnalysisText : 'None'}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-500 font-medium">Status</span>
                  <span className={`px-3 py-1.5 ${monitoringStatusColor} rounded-xl text-xs font-bold border-2`}>
                    {monitoringStatus}
                  </span>
                </div>
              </div>
            </div>

            <div className="bg-primary-500 rounded-2xl p-6 text-white shadow-lg animate-fade-in-up" style={{ animationDelay: '600ms' }}>
              <h3 className="text-xl font-bold mb-4">Next Steps</h3>
              <p className="text-white/90 text-sm mb-6 leading-relaxed">
                Based on your recent analyses, {monitoringStatus === 'Monitoring' ? 'continue regular monitoring every 6 months.' : 'we recommend consulting your doctor for further review.'}
              </p>
              <Link
                to="/new-analysis"
                className="bg-white text-primary-600 px-5 py-3 rounded-xl text-sm font-bold hover:bg-gray-50 transition-all duration-300 transform hover:scale-105 inline-flex items-center shadow-lg"
              >
                <Upload className="h-4 w-4 mr-2" />
                Start New Analysis
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PatientDashboard;
