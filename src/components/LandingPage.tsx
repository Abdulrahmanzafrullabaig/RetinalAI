import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Eye, Shield, Brain, Download, Users, Activity, LogOut, User } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

const LandingPage = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = async () => {
    await logout();
    navigate('/');
  };

  const getDashboardPath = () => {
    return user?.role === 'doctor' ? '/doctor-dashboard' : '/patient-dashboard';
  };
  return (
    <div className="min-h-screen bg-background-100">
      {/* Navigation */}
      <nav className="bg-primary-500 shadow-sm border-b border-primary-600">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-2">
              <Eye className="h-8 w-8 text-white" />
              <span className="text-xl font-bold text-white">RetinalAI</span>
            </div>
            <div className="flex items-center space-x-4">
              {user ? (
                // Authenticated user navigation
                <>
                  <div className="flex items-center space-x-3">
                    <div className="flex items-center space-x-2 text-white">
                      <User className="h-5 w-5" />
                      <span className="text-sm font-medium text-gold-400">
                        {user.username || user.full_name}
                      </span>
                      <span className="text-xs bg-primary-100 text-primary-600 px-2 py-1 rounded-full capitalize">
                        {user.role}
                      </span>
                    </div>
                    <Link 
                      to={getDashboardPath()}
                      className="bg-gold-500 hover:bg-gold-600 text-white px-4 py-2 rounded-md text-sm font-medium transition-colors shadow-sm"
                    >
                      Dashboard
                    </Link>
                    <button
                      onClick={handleLogout}
                      className="flex items-center space-x-1 text-white hover:text-gold-300 px-3 py-2 rounded-md text-sm font-medium transition-colors"
                    >
                      <LogOut className="h-4 w-4" />
                      <span>Logout</span>
                    </button>
                  </div>
                </>
              ) : (
                // Non-authenticated user navigation
                <>
                  <Link 
                    to="/login" 
                      className="text-white hover:text-gold-300 px-3 py-2 rounded-md text-sm font-medium transition-colors"
                  >
                    Login
                  </Link>
                  <Link 
                    to="/register" 
                    className="bg-gold-500 hover:bg-gold-600 text-white px-4 py-2 rounded-md text-sm font-medium transition-colors shadow-sm"
                  >
                    Get Started
                  </Link>
                </>
              )}
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="bg-gradient-to-br from-primary-600 to-primary-700 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div>
              <h1 className="text-4xl lg:text-5xl font-bold mb-6 leading-tight">
                {user ? 
                  <span className="text-gold-400">Welcome back, {user.username || user.full_name}!</span> : 
                  <span className="text-gold-400">Advanced Diabetic Retinopathy Detection with AI</span>
                }
              </h1>
              <p className="text-xl mb-8 text-primary-100">
                {user ? 
                  `Access your ${user.role} dashboard to manage your retinal health analysis and get AI-powered insights for better eye care.` :
                  'Get accurate, AI-powered analysis of your retinal images with explainable results and personalized recommendations for better eye health.'
                }
              </p>
              <div className="flex flex-col sm:flex-row gap-4">
                {user ? (
                  <>
                    <Link 
                      to={getDashboardPath()}
                      className="bg-gold-500 hover:bg-gold-600 text-white px-8 py-4 rounded-lg font-semibold text-lg transition-colors shadow-lg inline-flex items-center justify-center"
                    >
                      Go to Dashboard
                    </Link>
                    {user.role === 'patient' && (
                      <Link 
                        to="/new-analysis" 
                        className="border-2 border-white text-white hover:bg-gold-500 hover:border-gold-500 px-8 py-4 rounded-lg font-semibold text-lg transition-colors inline-flex items-center justify-center"
                      >
                        New Analysis
                      </Link>
                    )}
                  </>
                ) : (
                  <>
                    <Link 
                      to="/register" 
                      className="bg-gold-500 hover:bg-gold-600 text-white px-8 py-4 rounded-lg font-semibold text-lg transition-colors shadow-lg inline-flex items-center justify-center"
                    >
                      Start Analysis
                    </Link>
                    <Link 
                      to="#how-it-works" 
                      className="border-2 border-white text-white hover:bg-gold-500 hover:border-gold-500 px-8 py-4 rounded-lg font-semibold text-lg transition-colors inline-flex items-center justify-center"
                    >
                      Learn More
                    </Link>
                  </>
                )}
              </div>
            </div>
            <div className="relative">
              <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-8 border border-white/20">
                <div className="grid grid-cols-2 gap-4 mb-6">
                  <div className="bg-white/20 rounded-lg p-4 text-center">
                    <Brain className="h-8 w-8 mx-auto mb-2" />
                    <span className="text-sm font-medium">4 AI Models</span>
                  </div>
                  <div className="bg-white/20 rounded-lg p-4 text-center">
                    <Shield className="h-8 w-8 mx-auto mb-2" />
                    <span className="text-sm font-medium">Explainable AI</span>
                  </div>
                  <div className="bg-white/20 rounded-lg p-4 text-center">
                    <Activity className="h-8 w-8 mx-auto mb-2" />
                    <span className="text-sm font-medium">Real-time</span>
                  </div>
                  <div className="bg-white/20 rounded-lg p-4 text-center">
                    <Download className="h-8 w-8 mx-auto mb-2" />
                    <span className="text-sm font-medium">PDF Reports</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Information Section */}
      <section className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl lg:text-4xl font-bold text-primary-800 mb-4">
              Understanding Diabetic Retinopathy
            </h2>
            <p className="text-xl text-primary-600 max-w-3xl mx-auto">
              Early detection and prevention are key to preserving vision. Our AI system identifies all stages of diabetic retinopathy with precision.
            </p>
          </div>
          
          <div className="grid lg:grid-cols-5 gap-8">
            {[
              { stage: 'No DR', color: 'bg-green-100 border-green-200', icon: '✓', description: 'No signs of diabetic retinopathy detected' },
              { stage: 'Mild', color: 'bg-yellow-100 border-yellow-200', icon: '⚠', description: 'Microaneurysms present, regular monitoring needed' },
              { stage: 'Moderate', color: 'bg-orange-100 border-orange-200', icon: '⚡', description: 'Blood vessels blocked, closer monitoring required' },
              { stage: 'Severe', color: 'bg-red-100 border-red-200', icon: '⚠', description: 'Many blood vessels blocked, treatment needed' },
              { stage: 'Proliferative', color: 'bg-purple-100 border-purple-200', icon: '🚨', description: 'New abnormal blood vessels, urgent treatment' }
            ].map((stage, index) => (
              <div key={index} className={`${stage.color} border-2 rounded-xl p-6 text-center transition-transform hover:scale-105`}>
                <div className="text-3xl mb-3">{stage.icon}</div>
                <h3 className="text-lg font-bold text-primary-800 mb-2">{stage.stage}</h3>
                <p className="text-sm text-primary-700">{stage.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="bg-background-100 py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl lg:text-4xl font-bold text-primary-800 mb-4">
              How It Works
            </h2>
            <p className="text-xl text-primary-600 max-w-3xl mx-auto">
              Our advanced AI system uses multiple deep learning models to provide accurate and explainable results.
            </p>
          </div>
          
          <div className="grid lg:grid-cols-4 gap-8">
            {[
              { step: '1', title: 'Upload Image', description: 'Upload your retinal fundus image securely', icon: '📤' },
              { step: '2', title: 'AI Analysis', description: 'Four AI models analyze your image simultaneously', icon: '🧠' },
              { step: '3', title: 'Majority Voting', description: 'Advanced algorithm determines final diagnosis', icon: '🗳️' },
              { step: '4', title: 'Explainable Results', description: 'Get detailed explanations and recommendations', icon: '📊' }
            ].map((step, index) => (
              <div key={index} className="text-center group">
                <div className="bg-primary-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:bg-primary-200 transition-colors">
                  <span className="text-2xl">{step.icon}</span>
                </div>
                <div className="bg-primary-600 text-white rounded-full w-8 h-8 flex items-center justify-center mx-auto mb-3 text-sm font-bold">
                  {step.step}
                </div>
                <h3 className="text-lg font-bold text-black mb-2">{step.title}</h3>
                <p className="text-black">{step.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-primary-800 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid lg:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center space-x-2 mb-4">
                <Eye className="h-6 w-6 text-white" />
                <span className="text-xl font-bold text-white">RetinalAI</span>
              </div>
              <p className="text-primary-200">
                Advanced diabetic retinopathy detection powered by artificial intelligence.
              </p>
            </div>
            <div>
              <h4 className="text-lg font-semibold mb-4 text-white">Quick Links</h4>
              <ul className="space-y-2 text-primary-200">
                <li><Link to="/login" className="text-white hover:text-gold-300 transition-colors">Login</Link></li>
                <li><Link to="/register" className="text-white hover:text-gold-300 transition-colors">Register</Link></li>
                <li><a href="#how-it-works" className="text-white hover:text-gold-300 transition-colors">How It Works</a></li>
              </ul>
            </div>
            <div>
              <h4 className="text-lg font-semibold mb-4 text-white">For Healthcare</h4>
              <ul className="space-y-2 text-primary-200">
                <li><span className="text-white hover:text-gold-300 transition-colors">Doctor Dashboard</span></li>
                <li><span className="text-white hover:text-gold-300 transition-colors">Patient Management</span></li>
                <li><span className="text-white hover:text-gold-300 transition-colors">Report Analysis</span></li>
              </ul>
            </div>
            <div>
              <h4 className="text-lg font-semibold mb-4 text-white">Contact</h4>
              <div className="text-primary-200 space-y-2">
                <p className="text-white">Abdulrahmanzafrullabaig@gmail.com</p>
                <p className="text-white">+91-9731303697</p>
                <p className="text-white">Available 24/7</p>
              </div>
            </div>
          </div>
          <div className="border-t border-primary-700 mt-8 pt-8 text-center text-primary-200">
            <p className="text-white">&copy; 2025 RetinalAI. All rights reserved. This is a demonstration system.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;