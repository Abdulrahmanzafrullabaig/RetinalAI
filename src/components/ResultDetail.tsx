import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, Download, Share, Calendar, Brain, Eye, AlertCircle, CheckCircle, RefreshCcw, Lightbulb, Activity } from 'lucide-react';
import Header from './Header';

interface ModelPrediction {
  prediction: string;
  confidence: string;
  error?: string;
}

interface ExplanationImages {
  gradcam: { [key: string]: string | null };    // Model name -> Base64 image string or null
}

interface ReportDetail {
  id: number;
  filename: string;
  final_result: string;
  confidence: string;
  date: string;
  doctor_notes?: string;
  image: string; // URL to the uploaded image
  models: { [key: string]: ModelPrediction };
  explanations: ExplanationImages;
  risk_factors: string[];
  recommendations: string[];
}

interface Doctor {
  id: number;
  name: string;
  email: string;
}

const ResultDetail = () => {
  const { id } = useParams<{ id: string }>();
  const [result, setResult] = useState<ReportDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [doctors, setDoctors] = useState<Doctor[]>([]);
  const [selectedDoctor, setSelectedDoctor] = useState('');
  const [sharing, setSharing] = useState(false);
  const [shareError, setShareError] = useState('');
  const [showInsights, setShowInsights] = useState(false);
  const [insights, setInsights] = useState('');
  const [loadingInsights, setLoadingInsights] = useState(false);
  const [insightsError, setInsightsError] = useState('');

  useEffect(() => {
    const fetchResult = async () => {
      setLoading(true);
      setError('');
      try {
        const response = await fetch(`/api/result/${id}`, {
          method: 'GET',
          credentials: 'include',
        });
        if (response.ok) {
          const data: ReportDetail = await response.json();
          setResult(data);
        } else {
          const errorData = await response.json();
          setError(errorData.message || 'Failed to load result');
        }
      } catch (err) {
        console.error('Error fetching result:', err);
        setError('Error fetching result. Please check your connection.');
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

    fetchResult();
    fetchDoctors();
  }, [id]);

  const getResultColor = (resultText: string) => {
    if (resultText === 'None' || resultText === 'Not a fundus image') {
      return 'text-gray-600 bg-gray-50 border-gray-200';
    }
    const shortResult = resultText.split(' - ')[0];
    switch (shortResult) {
      case 'No DR': return 'text-green-600 bg-green-50 border-green-200';
      case 'Mild DR': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'Moderate DR': return 'text-orange-600 bg-orange-50 border-orange-200';
      case 'Severe DR': return 'text-red-600 bg-red-50 border-red-200';
      case 'Proliferative DR': return 'text-purple-600 bg-purple-50 border-purple-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getModelIcon = (model: string) => {
    const icons: { [key: string]: string } = {
      resnet50: '🔥',
      efficientnet: '⚡',
      vgg16: '🧠',
      mobilenetv4: '📱',
    };
    return icons[model] || '🤖';
  };

  const handleDownload = async () => {
    if (!result) return;
    try {
      const response = await fetch(`/api/download-report/${result.id}`, {
        method: 'GET',
        credentials: 'include',
      });
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `report_${result.id}.pdf`;
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

  const handleShare = async () => {
    if (!selectedDoctor || !result) {
      setShareError('Please select a doctor to share with.');
      return;
    }
    setSharing(true);
    setShareError('');
    try {
      const response = await fetch('/api/share-report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ report_id: result.id, doctor_id: selectedDoctor }),
      });
      const data = await response.json();
      if (response.ok && data.success) {
        // alert('Report shared successfully'); // Replacing alert
        console.log('Report shared successfully:', data.message);
        setSelectedDoctor(''); // Clear selected doctor after successful share
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

  const handleGetInsights = async () => {
    if (!result) return;
    
    setLoadingInsights(true);
    setInsightsError('');
    
    try {
      const response = await fetch('/api/ai-insights', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          prediction: result.final_result,
          confidence: result.confidence,
          user_context: {
            has_reports: true,
            report_id: result.id
          }
        })
      });
      
      const data = await response.json();
      
      if (response.ok && data.success) {
        setInsights(data.insights);
        setShowInsights(true);
      } else {
        setInsightsError(data.message || 'Failed to generate insights');
      }
    } catch (err) {
      console.error('Error getting AI insights:', err);
      setInsightsError('Error getting insights. Please check your connection.');
    } finally {
      setLoadingInsights(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-background-100 flex items-center justify-center">
        <div className="flex items-center space-x-2 text-primary-700">
          <RefreshCcw className="h-5 w-5 animate-spin" />
          <span className="font-body">Loading Report Details...</span>
        </div>
      </div>
    );
  }

  if (error || !result) {
    return (
      <div className="min-h-screen bg-background-100 flex items-center justify-center">
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg relative" role="alert">
          <strong className="font-heading font-bold">Error!</strong>
          <span className="block sm:inline font-body"> {error || 'Result not found.'}</span>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background-100">
      <Header />
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-8">
          <div className="flex items-center space-x-4 mb-4 sm:mb-0">
            <Link
              to="/results"
              className="p-2 text-primary-600 hover:text-primary-800 hover:bg-primary-100 rounded-lg transition-colors"
            >
              <ArrowLeft className="h-5 w-5" />
            </Link>
            <div>
              <h1 className="text-3xl font-heading font-bold text-primary-900">Analysis Report #{result.id}</h1>
              <p className="text-primary-600 font-body flex items-center space-x-2 mt-1">
                <Calendar className="h-4 w-4" />
                <span>{result.date}</span>
              </p>
            </div>
          </div>
          <div className="flex flex-col sm:flex-row items-start sm:items-center space-y-3 sm:space-y-0 sm:space-x-3 w-full sm:w-auto">
            <div className="relative w-full sm:w-auto">
              <select
                value={selectedDoctor}
                onChange={(e) => setSelectedDoctor(e.target.value)}
                className="block w-full px-4 py-2 border border-primary-200 rounded-lg text-primary-700 font-body focus:ring-2 focus:ring-primary-500"
              >
                <option value="">Select a doctor</option>
                {doctors.map((doctor) => (
                  <option key={doctor.id} value={doctor.id}>
                    {doctor.name}
                  </option>
                ))}
              </select>
              {shareError && (
                <p className="text-red-600 text-sm mt-1">{shareError}</p>
              )}
            </div>
            <button
              onClick={handleShare}
              disabled={sharing || !selectedDoctor}
              className="flex items-center space-x-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white font-heading rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed w-full sm:w-auto justify-center"
            >
              <Share className="h-4 w-4" />
              <span>{sharing ? 'Sharing...' : 'Share with Doctor'}</span>
            </button>
            <button
              onClick={handleDownload}
              className="flex items-center space-x-2 px-4 py-2 bg-gold-500 hover:bg-gold-600 text-white font-heading rounded-lg transition-colors w-full sm:w-auto justify-center"
            >
              <Download className="h-4 w-4" />
              <span>Download PDF</span>
            </button>
          </div>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-white rounded-2xl shadow-sm border border-primary-100 p-6">
              <h3 className="text-lg font-heading font-semibold text-primary-900 mb-4">Analysis Summary</h3>
              <div className="grid lg:grid-cols-2 gap-6">
                <div>
                  <p className="text-primary-700 font-body mb-2">Diagnosis:</p>
                  <span className={`px-3 py-1 rounded-full text-sm font-heading font-medium border ${getResultColor(result.final_result)}`}>
                    {result.final_result}
                  </span>
                  <p className="text-primary-700 font-body mt-2">Confidence:</p>
                  <p className="text-primary-900 font-body font-medium">{result.confidence}</p>
                  <p className="text-primary-700 font-body mt-2">Analysis Date:</p>
                  <p className="text-primary-900 font-body font-medium">{result.date}</p>
                  
                  {/* AI Insights Button */}
                  <div className="mt-6">
                    <button
                      onClick={handleGetInsights}
                      disabled={loadingInsights}
                      className="w-full flex items-center justify-center space-x-2 px-4 py-3 bg-gold-500 hover:bg-gold-600 text-white rounded-lg font-heading font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105 shadow-lg"
                    >
                      {loadingInsights ? (
                        <>
                          <RefreshCcw className="h-5 w-5 animate-spin" />
                          <span>Generating Insights...</span>
                        </>
                      ) : (
                        <>
                          <Lightbulb className="h-5 w-5" />
                          <span>Get AI Insights & Recommendations</span>
                        </>
                      )}
                    </button>
                    
                    {insightsError && (
                      <div className="mt-3 bg-red-50 border border-red-200 text-red-700 px-3 py-2 rounded-lg text-sm">
                        {insightsError}
                      </div>
                    )}
                  </div>
                </div>
                <div className="aspect-square flex items-center justify-center bg-gray-50 rounded-lg overflow-hidden">
                  <img
                    src={`http://localhost:5000${result.image}`}
                    alt="Fundus Image"
                    className="w-full h-full object-contain"
                    onError={(e) => { 
                      const target = e.currentTarget as HTMLImageElement;
                      target.src = 'https://placehold.co/200x200/cccccc/ffffff?text=Image+Not+Found'; 
                      target.onerror = null; 
                    }}
                  />
                </div>
              </div>
            </div>

            <div className="bg-white rounded-2xl shadow-sm border border-primary-100 p-6">
              <h3 className="text-lg font-heading font-semibold text-primary-900 mb-4">Model Predictions</h3>
              <div className="grid grid-cols-2 gap-4">
                {Object.entries(result.models).map(([model, prediction]) => (
                  <div key={model} className="bg-background-100 rounded-lg p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <span className="text-lg">{getModelIcon(model)}</span>
                      <p className="text-primary-800 font-heading font-medium capitalize">{model.replace('_', ' ')}</p>
                    </div>
                    {prediction.error ? (
                      <p className="text-sm font-body text-red-700">Error: {prediction.error}</p>
                    ) : (
                      <>
                        <p className="text-sm font-body text-primary-700">Prediction: <span className={`font-medium ${getResultColor(prediction.prediction)}`}>{prediction.prediction}</span></p>
                        <p className="text-sm font-body text-primary-700">Confidence: {prediction.confidence}</p>
                      </>
                    )}
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-white rounded-2xl shadow-sm border border-primary-100 p-6">
              <h3 className="text-lg font-heading font-semibold text-primary-900 mb-4">Explainable AI Visualizations</h3>
              
              {/* Grad-CAM Section */}
              <div>
                <h4 className="text-md font-heading font-semibold text-primary-800 mb-4">Grad-CAM Analysis - Visual Attention Maps</h4>
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                  {Object.entries(result.explanations.gradcam || {}).map(([modelName, gradcamImage]) => (
                    <div key={`gradcam-${modelName}`} className="bg-background-100 rounded-lg p-3">
                      <h5 className="text-sm font-heading font-medium text-primary-700 mb-2 capitalize">{modelName.replace('_', ' ')}</h5>
                      {gradcamImage ? (
                        <img
                          src={`data:image/png;base64,${gradcamImage}`}
                          alt={`Grad-CAM - ${modelName}`}
                          className="w-full rounded-lg aspect-square object-cover"
                        />
                      ) : (
                        <div className="bg-gradient-to-br from-primary-100 to-gold-100 rounded-lg p-4 aspect-square flex items-center justify-center">
                          <div className="text-center">
                            <Brain className="h-8 w-8 text-gold-600 mx-auto mb-1" />
                            <p className="text-xs font-body text-gold-700">N/A</p>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
              
              <p className="text-sm font-body text-primary-600 mt-4 text-center">
                Visualizations show attention maps highlighting the areas each AI model focused on for diagnosis
              </p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-white rounded-2xl shadow-sm border border-primary-100 p-6">
              <h3 className="text-lg font-heading font-semibold text-primary-900 mb-4">Risk Assessment</h3>
              <div className="space-y-3">
                {result.risk_factors && result.risk_factors.length > 0 ? (
                  result.risk_factors.map((factor: string, index: number) => (
                    <div key={index} className="flex items-start space-x-2">
                      <AlertCircle className="h-5 w-5 text-amber-500 mt-0.5 flex-shrink-0" />
                      <p className="text-sm font-body text-primary-700">{factor}</p>
                    </div>
                  ))
                ) : (
                  <p className="text-sm font-body text-primary-700">No specific risk factors identified</p>
                )}
              </div>
            </div>

            <div className="bg-white rounded-2xl shadow-sm border border-primary-100 p-6">
              <h3 className="text-lg font-heading font-semibold text-primary-900 mb-4">AI Recommendations</h3>
              <div className="space-y-3">
                {result.recommendations.map((recommendation: string, index: number) => (
                  <div key={index} className="flex items-start space-x-2">
                    <CheckCircle className="h-5 w-5 text-green-500 mt-0.5 flex-shrink-0" />
                    <p className="text-sm font-body text-primary-700">{recommendation}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-gradient-to-br from-primary-500 to-gold-500 rounded-2xl p-6 text-white">
              <h3 className="text-lg font-heading font-semibold mb-4">Next Steps</h3>
              <div className="space-y-3">
                <Link
                  to="/appointments"
                  className="block w-full bg-white text-primary-600 px-4 py-2 rounded-lg font-heading font-medium hover:bg-primary-50 transition-colors text-center"
                >
                  Book Appointment
                </Link>
                <Link
                  to="/predict"
                  className="block w-full border border-primary-300 text-white px-4 py-2 rounded-lg font-heading font-medium hover:bg-primary-600 transition-colors text-center"
                >
                  New Analysis
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* AI Insights Modal */}
      {showInsights && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-2xl shadow-xl max-w-5xl w-full max-h-[85vh] overflow-hidden">
            <div className="bg-primary-500 text-white p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <Lightbulb className="h-7 w-7" />
                  <div>
                    <h2 className="text-2xl font-heading font-bold">AI Insights & Recommendations</h2>
                    <p className="text-primary-100 font-body">Personalized analysis for your diagnosis</p>
                  </div>
                </div>
                <button
                  onClick={() => setShowInsights(false)}
                  className="text-white hover:bg-white hover:bg-opacity-20 rounded-full p-2 transition-all"
                >
                  ✕
                </button>
              </div>
            </div>
            
            <div className="p-8 overflow-y-auto max-h-[calc(85vh-140px)] font-body">
              <div className="mb-8 bg-background-100 rounded-xl p-6 border border-primary-200">
                <div className="flex items-center space-x-3 mb-4">
                  <Activity className="h-6 w-6 text-primary-600" />
                  <h3 className="font-heading text-xl font-bold text-primary-900">Your Diagnosis</h3>
                </div>
                <div className="flex flex-col sm:flex-row items-start sm:items-center space-y-2 sm:space-y-0 sm:space-x-4">
                  <span className={`px-4 py-2 rounded-full text-sm font-heading font-medium border ${getResultColor(result?.final_result || '')}`}>
                    {result?.final_result}
                  </span>
                  <span className="text-primary-700 font-body font-medium">Confidence: {result?.confidence}</span>
                </div>
              </div>
              
              <div className="space-y-8">
                <div className="prose prose-lg max-w-none">
                  {insights.split('\n').map((line, index) => {
                    // Clean up the line by removing markdown symbols
                    const cleanLine = line
                      .replace(/#{1,6}\s*/g, '') // Remove heading markdown
                      .replace(/\*\*\*/g, '') // Remove triple asterisks
                      .replace(/\*\*/g, '') // Remove double asterisks
                      .replace(/^\s*[-*+]\s+/g, '• '); // Convert list markers to bullet points
                    
                    if (cleanLine.startsWith('WHY THIS PREDICTION') || 
                        cleanLine.startsWith('PREVENTION STRATEGIES') ||
                        cleanLine.startsWith('PRECAUTIONS') ||
                        cleanLine.startsWith('PERSONALIZED SUGGESTIONS') ||
                        cleanLine.startsWith('MEDICAL DISCLAIMER')) {
                      const title = cleanLine.trim();
                      return (
                        <div key={index} className="mb-6 mt-8 first:mt-0">
                          <h3 className="text-2xl font-heading font-bold text-primary-900 flex items-center space-x-3 mb-4 pb-2 border-b-2 border-primary-200">
                            {title.includes('WHY') && <Brain className="h-6 w-6 text-gold-500" />}
                            {title.includes('PREVENTION') && <CheckCircle className="h-6 w-6 text-green-600" />}
                            {title.includes('PRECAUTIONS') && <AlertCircle className="h-6 w-6 text-amber-500" />}
                            {title.includes('SUGGESTIONS') && <Lightbulb className="h-6 w-6 text-primary-600" />}
                            {title.includes('DISCLAIMER') && <Eye className="h-6 w-6 text-gray-600" />}
                            <span>{title}</span>
                          </h3>
                        </div>
                      );
                    } else if (cleanLine.startsWith('•')) {
                      return (
                        <div key={index} className="flex items-start space-x-3 mb-4 ml-6">
                          <div className="w-2 h-2 bg-gold-500 rounded-full mt-3 flex-shrink-0"></div>
                          <p className="text-gray-800 font-body leading-relaxed text-lg">{cleanLine.substring(1).trim()}</p>
                        </div>
                      );
                    } else if (cleanLine.trim()) {
                      return (
                        <div key={index} className="mb-6">
                          <p className="text-gray-800 font-body leading-relaxed text-lg">
                            {cleanLine}
                          </p>
                        </div>
                      );
                    }
                    return null;
                  })}
                </div>
              </div>
              
              <div className="mt-10 bg-primary-50 border-2 border-primary-200 rounded-xl p-6">
                <div className="flex items-center space-x-3 mb-4">
                  <Brain className="h-6 w-6 text-primary-600" />
                  <h4 className="font-heading text-lg font-bold text-primary-900">Powered by Advanced AI</h4>
                </div>
                <p className="text-primary-800 font-body leading-relaxed">
                  These insights are generated by Google's Gemini AI, specifically trained on medical knowledge and 
                  tailored to your diagnosis. This analysis combines the latest medical research with AI precision 
                  to provide you with actionable, personalized recommendations.
                </p>
              </div>
              
              <div className="mt-8 pt-6 border-t border-gray-200">
                <div className="flex flex-col lg:flex-row items-center justify-between space-y-4 lg:space-y-0 lg:space-x-6">
                  <div className="flex flex-col sm:flex-row items-center space-y-3 sm:space-y-0 sm:space-x-4 w-full lg:w-auto">
                    <Link
                      to="/appointments"
                      className="w-full sm:w-auto bg-gold-500 hover:bg-gold-600 text-white px-6 py-3 rounded-lg font-heading font-medium transition-colors text-center"
                    >
                      Book Appointment
                    </Link>
                    <Link
                      to="/predict"
                      className="w-full sm:w-auto border-2 border-primary-500 text-primary-600 hover:bg-primary-50 px-6 py-3 rounded-lg font-heading font-medium transition-colors text-center"
                    >
                      New Analysis
                    </Link>
                  </div>
                  <button
                    onClick={() => setShowInsights(false)}
                    className="text-gray-600 hover:text-gray-800 px-6 py-3 rounded-lg transition-colors font-heading font-medium"
                  >
                    Close Insights
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ResultDetail;
