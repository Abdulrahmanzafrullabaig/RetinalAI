import React from 'react';
import { Eye } from 'lucide-react';

const Footer = () => {
  return (
    <footer className="bg-primary-800 text-white py-8 mt-auto">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid lg:grid-cols-3 gap-8">
          <div>
            <div className="flex items-center space-x-2 mb-4">
              <Eye className="h-6 w-6" />
              <span className="text-xl font-bold">RetinalAI</span>
            </div>
            <p className="text-primary-200">
              Advanced diabetic retinopathy detection powered by artificial intelligence.
            </p>
          </div>
          <div>
            <h4 className="text-lg font-semibold mb-4">Quick Links</h4>
            <ul className="space-y-2 text-primary-200">
              <li><span className="hover:text-white transition-colors cursor-pointer">How It Works</span></li>
              <li><span className="hover:text-white transition-colors cursor-pointer">AI Models</span></li>
              <li><span className="hover:text-white transition-colors cursor-pointer">Privacy Policy</span></li>
            </ul>
          </div>
          <div>
            <h4 className="text-lg font-semibold mb-4">Contact</h4>
            <div className="text-primary-200 space-y-2">
              <p>Abdulrahmanzafrullabaig@gmail.com</p>
              <p>+91-9731303697</p>
              <p>Available 24/7</p>
            </div>
          </div>
        </div>
        <div className="border-t border-primary-700 mt-8 pt-8 text-center text-primary-200">
          <p>&copy; 2025 RetinalAI. All rights reserved. This is a demonstration system.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;