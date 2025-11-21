import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Upload, History, Share, Calendar, Eye, RefreshCcw } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import Header from './Header';

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
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [authChecked, setAuthChecked] = useState(false);

  const fetchPatientData = async () => {
    setLoading(true);
    setError('');
    
    console.log('Fetching patient data...');
    
    try {
      const response = await fetch('/api/results', {
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
    
    checkAuthAndFetch();
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

  const totalAnalyses = recentAnalyses.length;
  const lastAnalysisDate = recentAnalyses.length > 0 ? recentAnalyses[0].date : 'N/A';
  // Determine monitoring status - this is a simplified example
  const monitoringStatus = totalAnalyses > 0 && (recentAnalyses[0].result === 'No DR - No visible abnormalities' || recentAnalyses[0].result === 'Negative for Diabetic Retinopathy') ? 'Monitoring' : 'Needs Review';
  const monitoringStatusColor = monitoringStatus === 'Monitoring' ? 'bg-green-100 text-green-700' : 'bg-amber-100 text-amber-700';
  
  // Fix date calculation to show whole days
  const lastAnalysisText = lastAnalysisDate !== 'N/A' ? 
    `${Math.floor((new Date().getTime() - new Date(lastAnalysisDate).getTime()) / (1000 * 3600 * 24))} days ago` : 'N/A';


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
    <div className="min-h-screen bg-background-100">
      <Header />
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Welcome Section */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-primary-900 mb-2">
            Welcome back, {user?.username}
          </h1>
          <p className="text-primary-600">
            Track your retinal health with AI-powered analysis and monitoring.
          </p>
        </div>

        {/* Quick Actions */}
        <div className="grid lg:grid-cols-4 gap-6 mb-8">
          <Link
            to="/new-analysis"
            className="bg-white rounded-xl p-6 shadow-sm border border-primary-100 hover:shadow-md transition-all duration-200 group"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="bg-gold-500 rounded-lg p-3 group-hover:scale-110 transition-transform">
                <Upload className="h-6 w-6 text-white" />
              </div>
            </div>
            <h3 className="text-lg font-semibold text-primary-900 mb-2">New Analysis</h3>
            <p className="text-sm text-primary-600">Choose image or clinical data analysis</p>
          </Link>

          <Link
            to="/results"
            className="bg-white rounded-xl p-6 shadow-sm border border-primary-100 hover:shadow-md transition-all duration-200 group"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="bg-primary-500 rounded-lg p-3 group-hover:scale-110 transition-transform">
                <History className="h-6 w-6 text-white" />
              </div>
            </div>
            <h3 className="text-lg font-semibold text-primary-900 mb-2">Analysis History</h3>
            <p className="text-sm text-primary-600">View all your previous results</p>
          </Link>

          {/* This "Share Reports" is static, actual sharing handled on results page.
              Keeping the UI element as per request. */}
          <div className="bg-white rounded-xl p-6 shadow-sm border border-primary-100 hover:shadow-md transition-all duration-200 group cursor-pointer">
            <div className="flex items-center justify-between mb-4">
              <div className="bg-blue-500 rounded-lg p-3 group-hover:scale-110 transition-transform">
                <Share className="h-6 w-6 text-white" />
              </div>
            </div>
            <h3 className="text-lg font-semibold text-primary-900 mb-2">Share Reports</h3>
            <p className="text-sm text-primary-600">Share results with your doctor</p>
          </div>

          <Link
            to="/appointments"
            className="bg-white rounded-xl p-6 shadow-sm border border-primary-100 hover:shadow-md transition-all duration-200 group"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="bg-purple-500 rounded-lg p-3 group-hover:scale-110 transition-transform">
                <Calendar className="h-6 w-6 text-white" />
              </div>
            </div>
            <h3 className="text-lg font-semibold text-primary-900 mb-2">Appointments</h3>
            <p className="text-sm text-primary-600">Schedule follow-up appointments</p>
          </Link>
        </div>

        {/* Recent Analyses */}
        <div className="grid lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2">
            <div className="bg-white rounded-xl shadow-sm border border-primary-100 p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-primary-900">Recent Analyses</h2>
                <button
                  onClick={fetchPatientData}
                  disabled={loading}
                  className="text-primary-600 hover:text-primary-800 transition-colors disabled:opacity-50"
                  title="Refresh data"
                >
                  <RefreshCcw className={`h-5 w-5 ${loading ? 'animate-spin' : ''}`} />
                </button>
              </div>
              <div className="space-y-4">
                {recentAnalyses.length > 0 ? (
                  recentAnalyses.map((analysis) => (
                    <div key={analysis.id} className="flex items-center justify-between p-4 bg-primary-50 rounded-lg">
                      <div className="flex items-center space-x-4">
                        <div className="bg-primary-100 rounded-lg p-2">
                          <Eye className="h-5 w-5 text-primary-600" />
                        </div>
                        <div>
                          <p className="font-medium text-primary-900">{analysis.date}</p>
                          <p className="text-sm text-primary-600">
                            {analysis.confidence !== 'N/A' ? `Confidence: ${analysis.confidence}` : 'Clinical Analysis'}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-3">
                        <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getResultColor(analysis.result)}`}>
                          {analysis.result.length > 30 ? 
                            `${analysis.result.substring(0, 30)}...` : 
                            analysis.result
                          }
                        </span>
                        <Link
                          to={`/result/${analysis.id}`}
                          className="text-primary-600 hover:text-primary-800 text-sm font-medium whitespace-nowrap"
                        >
                          View Details
                        </Link>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-12">
                    <div className="bg-gray-100 rounded-full p-4 w-16 h-16 mx-auto mb-4">
                      <Eye className="h-8 w-8 text-gray-400 mx-auto" />
                    </div>
                    <h3 className="text-lg font-medium text-gray-900 mb-2">No analyses yet</h3>
                    <p className="text-gray-500 mb-4">Start your first retinal health analysis</p>
                    <Link
                      to="/new-analysis"
                      className="inline-flex items-center px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
                    >
                      <Upload className="h-4 w-4 mr-2" />
                      Start Analysis
                    </Link>
                  </div>
                )}
              </div>
              {totalAnalyses > 3 && (
                <div className="mt-6 text-center">
                  <Link
                    to="/results"
                    className="text-primary-600 hover:text-primary-800 font-medium"
                  >
                    View All Results →
                  </Link>
                </div>
              )}
            </div>
          </div>

          {/* Health Summary */}
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-sm border border-primary-100 p-6">
              <h3 className="text-lg font-semibold text-primary-900 mb-4">Health Summary</h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-primary-700">Total Analyses</span>
                  <span className="font-semibold text-primary-900">{totalAnalyses}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-primary-700">Last Analysis</span>
                  <span className="font-semibold text-primary-900 text-right text-sm">
                    {lastAnalysisDate !== 'N/A' ? lastAnalysisText : 'None'}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-primary-700">Status</span>
                  <span className={`px-2 py-1 ${monitoringStatusColor} rounded-full text-xs font-medium`}>
                    {monitoringStatus}
                  </span>
                </div>
              </div>
            </div>

            <div className="bg-gradient-to-br from-primary-500 to-primary-600 rounded-xl p-6 text-white">
              <h3 className="text-lg font-semibold mb-3">Next Steps</h3>
              <p className="text-primary-100 text-sm mb-4">
                Based on your recent analyses, {monitoringStatus === 'Monitoring' ? 'continue regular monitoring every 6 months.' : 'we recommend consulting your doctor for further review.'}
              </p>
              <Link
                to="/new-analysis"
                className="bg-white text-primary-600 px-4 py-2 rounded-lg text-sm font-medium hover:bg-primary-50 transition-colors inline-flex items-center"
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
