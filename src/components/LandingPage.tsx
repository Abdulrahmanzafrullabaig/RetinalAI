import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Eye, Shield, Brain, Download, Users, Activity, LogOut, User, Sparkles, ArrowRight, CheckCircle2, Zap, Target } from 'lucide-react';
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
    <div className="min-h-screen bg-white">
      {/* Navigation */}
      <nav className="bg-primary-500 shadow-lg border-b-2 border-primary-600 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-20">
            <Link to="/" className="flex items-center space-x-3 group transition-all duration-300 hover:scale-105">
              <div className="relative">
                <div className="absolute inset-0 bg-secondary-500 rounded-xl blur-md opacity-0 group-hover:opacity-50 transition-opacity duration-300"></div>
                <div className="relative bg-white rounded-xl p-2.5 shadow-lg transform transition-all duration-300 group-hover:rotate-6">
                  <Eye className="h-7 w-7 text-primary-500" />
                </div>
              </div>
              <div className="flex flex-col">
                <span className="text-2xl font-bold text-white tracking-tight group-hover:text-secondary-500 transition-colors duration-300">
                  RetinalAI
                </span>
                <span className="text-xs text-white/80 font-medium -mt-1">AI Healthcare</span>
              </div>
            </Link>
            <div className="flex items-center space-x-4">
              {user ? (
                <>
                  <div className="flex items-center space-x-3">
                    <div className="flex items-center space-x-2 text-white bg-primary-600 px-4 py-2 rounded-xl">
                      <User className="h-5 w-5" />
                      <span className="text-sm font-semibold">{user.username || user.full_name}</span>
                      <span className="text-xs bg-secondary-500 text-white px-2 py-1 rounded-full capitalize font-medium">
                        {user.role}
                      </span>
                    </div>
                    <Link
                      to={getDashboardPath()}
                      className="bg-secondary-500 hover:bg-secondary-600 text-white px-6 py-2.5 rounded-xl text-sm font-semibold transition-all duration-300 shadow-lg transform hover:scale-105 flex items-center space-x-2"
                    >
                      <span>Dashboard</span>
                      <ArrowRight className="h-4 w-4" />
                    </Link>
                    <button
                      onClick={handleLogout}
                      className="flex items-center space-x-2 text-white hover:text-secondary-500 px-4 py-2.5 rounded-xl text-sm font-semibold transition-all duration-300 hover:bg-primary-600"
                    >
                      <LogOut className="h-4 w-4" />
                      <span>Logout</span>
                    </button>
                  </div>
                </>
              ) : (
                <>
                  <Link
                    to="/login"
                    className="text-white hover:text-secondary-500 px-4 py-2.5 rounded-xl text-sm font-semibold transition-all duration-300 hover:bg-primary-600"
                  >
                    Login
                  </Link>
                  <Link
                    to="/register"
                    className="bg-secondary-500 hover:bg-secondary-600 text-white px-6 py-2.5 rounded-xl text-sm font-semibold transition-all duration-300 shadow-lg transform hover:scale-105 flex items-center space-x-2"
                  >
                    <span>Get Started</span>
                    <ArrowRight className="h-4 w-4" />
                  </Link>
                </>
              )}
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="bg-primary-500 text-white relative overflow-hidden">
        <div className="absolute inset-0 opacity-10">
          <div className="absolute top-20 left-20 w-72 h-72 bg-secondary-500 rounded-full blur-3xl animate-pulse-slow"></div>
          <div className="absolute bottom-20 right-20 w-96 h-96 bg-white rounded-full blur-3xl animate-pulse-slow" style={{ animationDelay: '1s' }}></div>
        </div>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 lg:py-16 relative z-10">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <div className="animate-fade-in-up">
              <div className="inline-flex items-center space-x-2 bg-white/20 backdrop-blur-sm px-4 py-2 rounded-full mb-6 animate-slide-in-left">
                <Sparkles className="h-4 w-4 text-secondary-500" />
                <span className="text-sm font-semibold">AI-Powered Healthcare</span>
              </div>
              <h1 className="text-5xl lg:text-6xl font-bold mb-6 leading-tight">
                {user ? (
                  <span className="text-white">
                    Welcome back, <span className="text-secondary-500">{user.username || user.full_name}!</span>
                  </span>
                ) : (
                  <>
                    <span className="text-white">Advanced Diabetic</span>
                    <br />
                    <span className="text-secondary-500">Retinopathy Detection</span>
                    <br />
                    <span className="text-white">with AI</span>
                  </>
                )}
              </h1>
              <p className="text-xl lg:text-2xl mb-10 text-white/90 leading-relaxed">
                {user
                  ? `Access your ${user.role} dashboard to manage your retinal health analysis and get AI-powered insights for better eye care.`
                  : 'Get accurate, AI-powered analysis of your retinal images with explainable results and personalized recommendations for better eye health.'}
              </p>
              <div className="flex flex-col sm:flex-row gap-4">
                {user ? (
                  <>
                    <Link
                      to={getDashboardPath()}
                      className="group bg-secondary-500 hover:bg-secondary-600 text-white px-8 py-4 rounded-xl font-bold text-lg transition-all duration-300 shadow-xl transform hover:scale-105 flex items-center justify-center space-x-2"
                    >
                      <span>Go to Dashboard</span>
                      <ArrowRight className="h-5 w-5 group-hover:translate-x-1 transition-transform" />
                    </Link>
                    {user.role === 'patient' && (
                      <Link
                        to="/new-analysis"
                        className="group border-2 border-white text-white hover:bg-white hover:text-primary-500 px-8 py-4 rounded-xl font-bold text-lg transition-all duration-300 transform hover:scale-105 flex items-center justify-center"
                      >
                        New Analysis
                      </Link>
                    )}
                  </>
                ) : (
                  <>
                    <Link
                      to="/register"
                      className="group bg-secondary-500 hover:bg-secondary-600 text-white px-8 py-4 rounded-xl font-bold text-lg transition-all duration-300 shadow-xl transform hover:scale-105 flex items-center justify-center space-x-2"
                    >
                      <span>Start Analysis</span>
                      <ArrowRight className="h-5 w-5 group-hover:translate-x-1 transition-transform" />
                    </Link>
                    <Link
                      to="#how-it-works"
                      className="group border-2 border-white text-white hover:bg-white hover:text-primary-500 px-8 py-4 rounded-xl font-bold text-lg transition-all duration-300 transform hover:scale-105 flex items-center justify-center"
                    >
                      Learn More
                    </Link>
                  </>
                )}
              </div>
            </div>
            <div className="relative animate-fade-in-up" style={{ animationDelay: '200ms' }}>
              <div className="bg-white/10 backdrop-blur-md rounded-3xl p-8 border-2 border-white/20 shadow-2xl">
                <div className="grid grid-cols-2 gap-6 mb-6">
                  {[
                    { icon: Brain, label: '4 AI Models', color: 'bg-secondary-500' },
                    { icon: Shield, label: 'Explainable AI', color: 'bg-primary-600' },
                    { icon: Activity, label: 'Real-time', color: 'bg-secondary-500' },
                    { icon: Download, label: 'PDF Reports', color: 'bg-primary-600' },
                  ].map((item, index) => (
                    <div
                      key={index}
                      className={`${item.color} rounded-2xl p-6 text-center transform transition-all duration-300 hover:scale-110 hover:rotate-2 shadow-lg`}
                      style={{ animationDelay: `${index * 100}ms` }}
                    >
                      <item.icon className="h-10 w-10 mx-auto mb-3 text-white" />
                      <span className="text-sm font-bold text-white">{item.label}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Information Section */}
      <section className="py-24 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-20 animate-fade-in-up">
            <h2 className="text-4xl lg:text-5xl font-bold text-gray-900 mb-6">
              Understanding Diabetic Retinopathy
            </h2>
            <p className="text-xl text-gray-500 max-w-3xl mx-auto leading-relaxed">
              Early detection and prevention are key to preserving vision. Our AI system identifies all stages of diabetic retinopathy with precision.
            </p>
          </div>

          <div className="grid lg:grid-cols-5 gap-6">
            {[
              { stage: 'No DR', color: 'bg-green-50 border-green-300', icon: CheckCircle2, description: 'No signs of diabetic retinopathy detected' },
              { stage: 'Mild', color: 'bg-secondary-50 border-secondary-300', icon: Target, description: 'Microaneurysms present, regular monitoring needed' },
              { stage: 'Moderate', color: 'bg-orange-50 border-orange-300', icon: Zap, description: 'Blood vessels blocked, closer monitoring required' },
              { stage: 'Severe', color: 'bg-red-50 border-red-300', icon: Shield, description: 'Many blood vessels blocked, treatment needed' },
              { stage: 'Proliferative', color: 'bg-purple-50 border-purple-300', icon: Activity, description: 'New abnormal blood vessels, urgent treatment' }
            ].map((stage, index) => (
              <div
                key={index}
                className={`${stage.color} border-2 rounded-2xl p-6 text-center transition-all duration-300 transform hover:scale-105 hover:shadow-xl animate-fade-in-up`}
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <div className="bg-white rounded-full p-3 w-16 h-16 mx-auto mb-4 flex items-center justify-center shadow-md">
                  <stage.icon className="h-8 w-8 text-gray-700" />
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-3">{stage.stage}</h3>
                <p className="text-sm text-gray-500 leading-relaxed">{stage.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="bg-gray-100 py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-20 animate-fade-in-up">
            <h2 className="text-4xl lg:text-5xl font-bold text-gray-900 mb-6">
              How It Works
            </h2>
            <p className="text-xl text-gray-500 max-w-3xl mx-auto leading-relaxed">
              Our advanced AI system uses multiple deep learning models to provide accurate and explainable results.
            </p>
          </div>

          <div className="grid lg:grid-cols-4 gap-8">
            {[
              { step: '1', title: 'Upload Image', description: 'Upload your retinal fundus image securely', icon: '📤', color: 'bg-primary-500' },
              { step: '2', title: 'AI Analysis', description: 'Four AI models analyze your image simultaneously', icon: '🧠', color: 'bg-secondary-500' },
              { step: '3', title: 'Majority Voting', description: 'Advanced algorithm determines final diagnosis', icon: '🗳️', color: 'bg-primary-500' },
              { step: '4', title: 'Explainable Results', description: 'Get detailed explanations and recommendations', icon: '📊', color: 'bg-secondary-500' }
            ].map((step, index) => (
              <div
                key={index}
                className="text-center group animate-fade-in-up"
                style={{ animationDelay: `${index * 150}ms` }}
              >
                <div className={`${step.color} w-20 h-20 rounded-2xl flex items-center justify-center mx-auto mb-6 transform transition-all duration-300 group-hover:scale-110 group-hover:rotate-6 shadow-lg`}>
                  <span className="text-3xl">{step.icon}</span>
                </div>
                <div className={`${step.color} text-white rounded-full w-10 h-10 flex items-center justify-center mx-auto mb-4 text-lg font-bold shadow-md transform transition-all duration-300 group-hover:scale-110`}>
                  {step.step}
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-3">{step.title}</h3>
                <p className="text-gray-500 leading-relaxed">{step.description}</p>
              </div>
            ))}
          </div>

          {/* Demo Video Section */}
          <div className="mt-20 animate-fade-in-up">
            <div className="max-w-4xl mx-auto">
              <h3 className="text-3xl font-bold text-gray-900 text-center mb-6">Watch Demo</h3>
              <p className="text-lg text-gray-500 text-center mb-8">
                See how our AI-powered system analyzes retinal images and provides accurate diagnoses.
              </p>
              <div className="rounded-2xl overflow-hidden shadow-2xl border-2 border-gray-200 bg-white">
                <iframe
                  className="w-full h-96"
                  src="https://drive.google.com/file/d/1CSs6T6BYcjdRepsUofcQ42EOUzXwJGOs/preview"
                  title="Demo Video"
                  allow="autoplay; encrypted-media"
                  allowFullScreen
                ></iframe>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-primary-800 text-white py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid lg:grid-cols-4 gap-12">
            <div className="animate-fade-in-up">
              <div className="flex items-center space-x-3 mb-6">
                <div className="bg-white rounded-xl p-2">
                  <Eye className="h-6 w-6 text-primary-500" />
                </div>
                <div>
                  <span className="text-2xl font-bold text-white">RetinalAI</span>
                  <p className="text-xs text-white/80">AI Healthcare</p>
                </div>
              </div>
              <p className="text-white/80 leading-relaxed">
                Advanced diabetic retinopathy detection powered by artificial intelligence.
              </p>
            </div>
            <div className="animate-fade-in-up" style={{ animationDelay: '100ms' }}>
              <h4 className="text-lg font-bold mb-6 text-white">Quick Links</h4>
              <ul className="space-y-3 text-white/80">
                <li><Link to="/login" className="hover:text-secondary-500 transition-colors duration-300 flex items-center space-x-2 group"><span>Login</span> <ArrowRight className="h-3 w-3 opacity-0 group-hover:opacity-100 group-hover:translate-x-1 transition-all" /></Link></li>
                <li><Link to="/register" className="hover:text-secondary-500 transition-colors duration-300 flex items-center space-x-2 group"><span>Register</span> <ArrowRight className="h-3 w-3 opacity-0 group-hover:opacity-100 group-hover:translate-x-1 transition-all" /></Link></li>
                <li><a href="#how-it-works" className="hover:text-secondary-500 transition-colors duration-300 flex items-center space-x-2 group"><span>How It Works</span> <ArrowRight className="h-3 w-3 opacity-0 group-hover:opacity-100 group-hover:translate-x-1 transition-all" /></a></li>
              </ul>
            </div>
            <div className="animate-fade-in-up" style={{ animationDelay: '200ms' }}>
              <h4 className="text-lg font-bold mb-6 text-white">For Healthcare</h4>
              <ul className="space-y-3 text-white/80">
                <li><span className="hover:text-secondary-500 transition-colors duration-300 cursor-pointer flex items-center space-x-2 group"><span>Doctor Dashboard</span> <ArrowRight className="h-3 w-3 opacity-0 group-hover:opacity-100 group-hover:translate-x-1 transition-all" /></span></li>
                <li><span className="hover:text-secondary-500 transition-colors duration-300 cursor-pointer flex items-center space-x-2 group"><span>Patient Management</span> <ArrowRight className="h-3 w-3 opacity-0 group-hover:opacity-100 group-hover:translate-x-1 transition-all" /></span></li>
                <li><span className="hover:text-secondary-500 transition-colors duration-300 cursor-pointer flex items-center space-x-2 group"><span>Report Analysis</span> <ArrowRight className="h-3 w-3 opacity-0 group-hover:opacity-100 group-hover:translate-x-1 transition-all" /></span></li>
              </ul>
            </div>
            <div className="animate-fade-in-up" style={{ animationDelay: '300ms' }}>
              <h4 className="text-lg font-bold mb-6 text-white">Contact</h4>
              <div className="text-white/80 space-y-3">
                <p className="hover:text-secondary-500 transition-colors duration-300">Abdulrahmanzafrullabaig@gmail.com</p>
                <p className="hover:text-secondary-500 transition-colors duration-300">+91-9731303697</p>
                <p className="text-secondary-500 font-semibold">Available 24/7</p>
              </div>
            </div>
          </div>
          <div className="border-t border-primary-700 mt-12 pt-8 text-center text-white/80">
            <p className="text-white">&copy; 2025 RetinalAI. All rights reserved. This is a demonstration system.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
