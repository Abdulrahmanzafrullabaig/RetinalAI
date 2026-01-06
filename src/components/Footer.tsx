import { Eye, ArrowRight, Mail, Phone, Clock } from 'lucide-react';
import { Link } from 'react-router-dom';

const Footer = () => {
  return (
    <footer className="bg-primary-800 text-white py-16 mt-auto">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid lg:grid-cols-4 gap-12">
          <div className="animate-fade-in-up">
            <div className="flex items-center space-x-3 mb-6">
              <div className="bg-white rounded-xl p-2">
                <Eye className="h-6 w-6 text-primary-500" />
              </div>
              <div>
                <span className="text-2xl font-bold text-white">RetinalAI</span>
                <p className="text-xs text-white">AI Healthcare</p>
              </div>
            </div>
            <p className="text-white leading-relaxed">
              Advanced diabetic retinopathy detection powered by artificial intelligence.
            </p>
          </div>
          <div className="animate-fade-in-up" style={{ animationDelay: '100ms' }}>
            <h4 className="text-lg font-bold mb-6 text-white">Quick Links</h4>
            <ul className="space-y-3">
              <li>
                <Link to="/" className="text-white hover:text-secondary-500 transition-colors duration-300 flex items-center space-x-2 group">
                  <span>Home</span>
                  <ArrowRight className="h-3 w-3 opacity-0 group-hover:opacity-100 group-hover:translate-x-1 transition-all" />
                </Link>
              </li>
              <li>
                <a href="#how-it-works" className="text-white hover:text-secondary-500 transition-colors duration-300 flex items-center space-x-2 group">
                  <span>How It Works</span>
                  <ArrowRight className="h-3 w-3 opacity-0 group-hover:opacity-100 group-hover:translate-x-1 transition-all" />
                </a>
              </li>
              <li>
                <Link to="/login" className="text-white hover:text-secondary-500 transition-colors duration-300 flex items-center space-x-2 group">
                  <span>Login</span>
                  <ArrowRight className="h-3 w-3 opacity-0 group-hover:opacity-100 group-hover:translate-x-1 transition-all" />
                </Link>
              </li>
              <li>
                <Link to="/register" className="text-white hover:text-secondary-500 transition-colors duration-300 flex items-center space-x-2 group">
                  <span>Register</span>
                  <ArrowRight className="h-3 w-3 opacity-0 group-hover:opacity-100 group-hover:translate-x-1 transition-all" />
                </Link>
              </li>
            </ul>
          </div>
          <div className="animate-fade-in-up" style={{ animationDelay: '200ms' }}>
            <h4 className="text-lg font-bold mb-6 text-white">For Healthcare</h4>
            <ul className="space-y-3">
              <li>
                <span className="text-white hover:text-secondary-500 transition-colors duration-300 cursor-pointer flex items-center space-x-2 group">
                  <span>Doctor Dashboard</span>
                  <ArrowRight className="h-3 w-3 opacity-0 group-hover:opacity-100 group-hover:translate-x-1 transition-all" />
                </span>
              </li>
              <li>
                <span className="text-white hover:text-secondary-500 transition-colors duration-300 cursor-pointer flex items-center space-x-2 group">
                  <span>Patient Management</span>
                  <ArrowRight className="h-3 w-3 opacity-0 group-hover:opacity-100 group-hover:translate-x-1 transition-all" />
                </span>
              </li>
              <li>
                <span className="text-white hover:text-secondary-500 transition-colors duration-300 cursor-pointer flex items-center space-x-2 group">
                  <span>Report Analysis</span>
                  <ArrowRight className="h-3 w-3 opacity-0 group-hover:opacity-100 group-hover:translate-x-1 transition-all" />
                </span>
              </li>
            </ul>
          </div>
          <div className="animate-fade-in-up" style={{ animationDelay: '300ms' }}>
            <h4 className="text-lg font-bold mb-6 text-white">Contact</h4>
            <div className="space-y-4">
              <div className="flex items-center space-x-3 group">
                <Mail className="h-5 w-5 text-white group-hover:text-secondary-500 transition-colors duration-300" />
                <a href="mailto:Abdulrahmanzafrullabaig@gmail.com" className="text-white hover:text-secondary-500 transition-colors duration-300 hover:underline">
                  Abdulrahmanzafrullabaig@gmail.com
                </a>
              </div>
              <div className="flex items-center space-x-3 group">
                <Phone className="h-5 w-5 text-white group-hover:text-secondary-500 transition-colors duration-300" />
                <a href="tel:+919731303697" className="text-white hover:text-secondary-500 transition-colors duration-300 hover:underline">
                  +91-9731303697
                </a>
              </div>
              <div className="flex items-center space-x-3">
                <Clock className="h-5 w-5 text-white" />
                <span className="font-semibold text-white">Available 24/7</span>
              </div>
            </div>
          </div>
        </div>
        <div className="border-t-2 border-primary-700 mt-12 pt-8 text-center">
          <p className="text-white">&copy; 2025 RetinalAI. All rights reserved. This is a demonstration system.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
