import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Eye, Calendar, Download, Share, RefreshCcw } from 'lucide-react';
import Header from './Header';

interface ExplanationImages {
  gradcam: { [key: string]: string | null };    // Model name -> Base64 image string or null
}

interface Report {
  id: number;
  filename: string;
  result: string;
  confidence: string;
  date: string;
  models: { [key: string]: { prediction: string; confidence: string; error?: string } };
  doctor_notes?: string;
  image: string; // URL to the image
  explanations?: ExplanationImages; // Optional for compatibility
}

interface Doctor {
  id: number;
  name: string;
  email: string;
}

const Results = () => {
  const [results, setResults] = useState<Report[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [doctors, setDoctors] = useState<Doctor[]>([]);
  const [selectedReportId, setSelectedReportId] = useState<number | null>(null);
  const [selectedDoctor, setSelectedDoctor] = useState('');
  const [sharing, setSharing] = useState(false);
  const [shareError, setShareError] = useState('');

  const fetchResults = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await fetch('/api/results', {
        method: 'GET',
        credentials: 'include',
      });
      if (response.ok) {
        const data: Report[] = await response.json();
        setResults(data);
      } else {
        const errorData = await response.json();
        setError(errorData.message || 'Failed to load results');
      }
    } catch (err) {
      console.error('Error fetching results:', err);
      setError('Error fetching results. Please check your connection.');
    } finally {
      setLoading(false);
    }
  };

  const fetchDoctors = async () => {
    try {
      const response = await fetch('/api/doctors', {
        method: 'GET',
        credentials: 'include',
      });
      if (response.ok) {
        const data: Doctor[] = await response.json();
        setDoctors(data);
      } else {
        const errorData = await response.json();
        console.error('Failed to load doctors:', errorData.message);
      }
    } catch (err) {
      console.error('Error fetching doctors:', err);
    }
  };

  useEffect(() => {
    fetchResults();
    fetchDoctors();
  }, []);

  const getResultColor = (result: string) => {
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

  const handleDownload = async (reportId: number) => {
    try {
      const response = await fetch(`/api/download-report/${reportId}`, {
        method: 'GET',
        credentials: 'include',
      });
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `report_${reportId}.pdf`;
        document.body.appendChild(a); // Append to body to make it clickable in all browsers
        a.click();
        document.body.removeChild(a); // Clean up
        window.URL.revokeObjectURL(url);
      } else {
        const errorData = await response.json();
        console.error('Failed to download report:', errorData.message);
        // alert('Failed to download report'); // Replacing alert
      }
    } catch (err) {
      console.error('Error downloading report:', err);
      // alert('Error downloading report'); // Replacing alert
    }
  };

  const handleShare = async (reportId: number) => {
    if (!selectedDoctor) {
      setShareError('Please select a doctor to share with');
      return;
    }
    setSharing(true);
    setShareError('');
    try {
      const response = await fetch('/api/share-report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ report_id: reportId, doctor_id: selectedDoctor }),
      });
      const data = await response.json();
      if (response.ok && data.success) {
        // alert('Report shared successfully'); // Replacing alert
        console.log('Report shared successfully:', data.message);
        setSelectedReportId(null);
        setSelectedDoctor('');
      } else {
        setShareError(data.message || 'Failed to share report');
      }
    } catch (err) {
      console.error('Error sharing report:', err);
      setShareError('Error sharing report. Please check your connection.');
    } finally {
      setSharing(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-background-100 flex items-center justify-center">
        <div className="flex items-center space-x-2 text-primary-700">
          <RefreshCcw className="h-5 w-5 animate-spin" />
          <span>Loading Results...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-background-100 flex items-center justify-center">
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg relative" role="alert">
          <strong className="font-bold">Error!</strong>
          <span className="block sm:inline"> {error}</span>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background-100">
      <Header />
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold text-primary-900 mb-2">Analysis History</h1>
            <p className="text-primary-600">Track your retinal health over time</p>
          </div>
          <Link
            to="/predict"
            className="bg-gold-500 hover:bg-gold-600 text-white px-6 py-3 rounded-lg font-semibold transition-colors inline-flex items-center space-x-2"
          >
            <Eye className="h-5 w-5" />
            <span>New Analysis</span>
          </Link>
        </div>

        <div className="space-y-4">
          {results.length > 0 ? (
            results.map((result) => (
              <div key={result.id} className="bg-white rounded-xl shadow-sm border border-primary-100 p-6 hover:shadow-md transition-shadow">
                <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between">
                  <div className="flex items-start space-x-6 mb-4 lg:mb-0">
                    <div className="bg-primary-100 rounded-lg p-3 relative overflow-hidden w-16 h-16">
                      <img
                        src={`http://localhost:5000${result.image}`}
                        alt="Fundus Thumbnail"
                        className="w-full h-full object-cover rounded"
                        onError={(e) => {
                          const target = e.currentTarget as HTMLImageElement;
                          const parent = target.parentElement;
                          if (parent) {
                            parent.innerHTML = '<div class="flex items-center justify-center w-full h-full"><svg class="h-6 w-6 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"></path></svg></div>';
                          }
                        }}
                      />
                    </div>
                    <div>
                      <p className="text-primary-900 font-medium">{result.date}</p>
                      <p className="text-sm text-primary-600">Confidence: {result.confidence}</p>
                      <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getResultColor(result.result)}`}>
                        {result.result}
                      </span>
                    </div>
                  </div>
                  <div className="flex flex-col sm:flex-row items-end sm:items-center space-y-2 sm:space-y-0 sm:space-x-2 w-full lg:w-auto">
                    {selectedReportId === result.id ? (
                      <div className="flex items-center space-x-2 w-full sm:w-auto">
                        <select
                          value={selectedDoctor}
                          onChange={(e) => setSelectedDoctor(e.target.value)}
                          className="flex-grow sm:flex-grow-0 px-3 py-1 border border-primary-200 rounded-lg text-primary-700 focus:ring-2 focus:ring-primary-500"
                        >
                          <option value="">Select a doctor</option>
                          {doctors.map((doctor) => (
                            <option key={doctor.id} value={doctor.id}>
                              {doctor.name}
                            </option>
                          ))}
                        </select>
                        <button
                          onClick={() => handleShare(result.id)}
                          disabled={sharing || !selectedDoctor}
                          className="px-3 py-1 bg-primary-600 hover:bg-primary-700 text-white rounded-lg text-sm font-medium disabled:opacity-50"
                        >
                          {sharing ? 'Sharing...' : 'Share'}
                        </button>
                        <button
                          onClick={() => {
                            setSelectedReportId(null);
                            setShareError(''); // Clear error on cancel
                          }}
                          className="px-3 py-1 text-gray-600 hover:text-gray-800"
                        >
                          Cancel
                        </button>
                      </div>
                    ) : (
                      <>
                        <button
                          onClick={() => setSelectedReportId(result.id)}
                          className="p-2 text-primary-600 hover:text-primary-800 hover:bg-white rounded-lg transition-colors"
                          title="Share Report"
                        >
                          <Share className="h-4 w-4" />
                        </button>
                        <button
                          onClick={() => handleDownload(result.id)}
                          className="p-2 text-primary-600 hover:text-primary-800 hover:bg-white rounded-lg transition-colors"
                          title="Download Report"
                        >
                          <Download className="h-4 w-4" />
                        </button>
                        <Link
                          to={`/result/${result.id}`}
                          className="bg-primary-100 hover:bg-primary-200 text-primary-700 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
                        >
                          View Details
                        </Link>
                      </>
                    )}
                  </div>
                </div>
                {shareError && selectedReportId === result.id && (
                  <p className="text-red-600 text-sm mt-2">{shareError}</p>
                )}
                <div className="lg:hidden mt-4 pt-4 border-t border-gray-100">
                  <p className="text-sm text-primary-700 font-medium mb-2">Individual Model Results:</p>
                  <div className="grid grid-cols-2 gap-2">
                    {Object.entries(result.models).map(([model, prediction]) => (
                      <div key={model} className="flex justify-between text-sm">
                        <span className="text-gray-600 capitalize">{model.replace('_', ' ')}:</span>
                        <span className={`font-medium ${getResultColor(prediction.prediction || '')}`}>
                          {prediction.prediction || 'N/A'}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div className="text-center py-8 text-gray-500">
              No analysis results found. <Link to="/predict" className="text-primary-600 hover:underline">Start a new analysis</Link>!
            </div>
          )}
        </div>

        <div className="mt-8 grid lg:grid-cols-3 gap-6">
          <div className="bg-white rounded-xl p-6 shadow-sm border border-primary-100">
            <h3 className="text-lg font-semibold text-primary-900 mb-2">Total Analyses</h3>
            <p className="text-3xl font-bold text-primary-700">{results.length}</p>
            <p className="text-sm text-primary-600 mt-1">Since you joined</p>
          </div>
          <div className="bg-white rounded-xl p-6 shadow-sm border border-primary-100">
            <h3 className="text-lg font-semibold text-primary-900 mb-2">Latest Result</h3>
            <div className="flex items-center space-x-2">
              <span className={`px-3 py-1 rounded-full text-sm font-medium border ${getResultColor(results[0]?.result || '')}`}>
                {results[0]?.result || 'N/A'}
              </span>
            </div>
            <p className="text-sm text-primary-600 mt-1">{results[0]?.date || 'N/A'}</p>
          </div>
          <div className="bg-white rounded-xl p-6 shadow-sm border border-primary-100">
            <h3 className="text-lg font-semibold text-primary-900 mb-2">Monitoring Status</h3>
            <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-700 border border-green-200">
              Active Monitoring
            </span>
            <p className="text-sm text-primary-600 mt-1">Regular check-ups recommended</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Results;
