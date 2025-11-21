import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, Eye, ArrowRight, Image as ImageIcon, AlertCircle, X as XIcon } from 'lucide-react';
import Header from './Header';
import { useAuth } from '../context/AuthContext';

const Predict = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();
  const { user } = useAuth();

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFile = (file: File) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
      setError(''); // Clear any previous error
    } else {
      setError('Please upload a valid image file (PNG, JPG, JPEG)');
      setSelectedImage(null);
      setImagePreview(null);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedImage) {
      setError('Please select an image to analyze');
      return;
    }

    setIsAnalyzing(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('file', selectedImage);

      const response = await fetch('/api/predict', {
        method: 'POST',
        credentials: 'include',
        body: formData,
      });

      const data = await response.json();

      if (response.ok && data.success) {
        if (data.report.id) {
          // Normal fundus image - navigate to results
          navigate(`/result/${data.report.id}`);
        } else {
          // Not a fundus image - show error message
          setError(data.report.recommendations[0] || 'Please upload a valid fundus/retinal image for analysis.');
        }
      } else {
        setError(data.message || 'Analysis failed. Please try again.');
      }
    } catch (err) {
      console.error('Error performing analysis:', err);
      setError('Error performing analysis. Please check your connection.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleClearImage = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="min-h-screen bg-background-100">
      <Header />
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="text-center mb-8">
          <div className="flex items-center justify-center space-x-2 mb-4">
            <Eye className="h-8 w-8 text-primary-600" />
            <h1 className="text-3xl font-bold text-primary-900">AI Retinal Analysis</h1>
          </div>
          <p className="text-primary-600 text-lg">
            Upload your fundus image for comprehensive diabetic retinopathy detection
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          <div className="bg-white rounded-2xl shadow-sm border border-primary-100 p-6">
            <h2 className="text-xl font-semibold text-primary-900 mb-6">Upload Fundus Image</h2>
            <div
              className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-all duration-200 ${
                dragActive
                  ? 'border-primary-500 bg-primary-50'
                  : selectedImage
                  ? 'border-green-300 bg-green-50'
                  : 'border-gray-300 hover:border-primary-400 hover:bg-primary-50'
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                className="hidden"
              />
              {imagePreview ? (
                <div className="flex flex-col items-center relative">
                  <button
                    type="button"
                    onClick={handleClearImage}
                    className="absolute top-0 right-0 m-2 p-1 rounded-full bg-white/80 hover:bg-white shadow border border-gray-200 text-gray-700 hover:text-red-600"
                    aria-label="Remove image"
                    title="Remove image"
                  >
                    <XIcon className="h-4 w-4" />
                  </button>
                  <img
                    src={imagePreview}
                    alt="Preview"
                    className="max-w-full max-h-64 rounded-lg mb-4 object-contain"
                  />
                  <p className="text-sm text-primary-600">{selectedImage?.name}</p>
                </div>
              ) : (
                <div>
                  <ImageIcon className="h-12 w-12 text-primary-500 mx-auto mb-4" />
                  <p className="text-primary-700 mb-2">
                    Drag and drop your image here or
                  </p>
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="text-primary-600 hover:text-primary-800 font-medium"
                  >
                    Browse files
                  </button>
                </div>
              )}
            </div>
            {error && (
              <div className="mt-4 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm">
                {error}
              </div>
            )}
            <button
              onClick={handleAnalyze}
              disabled={isAnalyzing || !selectedImage}
              className="w-full mt-6 px-4 py-3 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
            >
              {isAnalyzing ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Analyzing...
                </>
              ) : (
                <>
                  <ArrowRight className="h-5 w-5 mr-2" />
                  Analyze Image
                </>
              )}
            </button>
          </div>

          <div className="space-y-6">
            <div className="bg-white rounded-2xl shadow-sm border border-primary-100 p-6">
              <h3 className="text-lg font-semibold text-primary-900 mb-4">Analysis Process</h3>
              <div className="space-y-4">
                {[
                  { step: 1, title: 'Image Preprocessing', description: 'Standardizing and optimizing your image', active: selectedImage && !isAnalyzing },
                  { step: 2, title: 'Multi-Model Analysis', description: 'Four AI models analyze simultaneously', active: isAnalyzing },
                  { step: 3, title: 'Majority Voting', description: 'Determining consensus diagnosis', active: false },
                  { step: 4, title: 'Explainable Results', description: 'Generating visual explanations', active: false },
                ].map((item) => (
                  <div key={item.step} className={`flex items-start space-x-3 ${item.active ? 'opacity-100' : 'opacity-60'}`}>
                    <div className={`rounded-full w-6 h-6 flex items-center justify-center text-xs font-bold ${
                      item.active ? 'bg-primary-600 text-white' : 'bg-gray-200 text-gray-600'
                    }`}>
                      {item.step}
                    </div>
                    <div>
                      <p className={`font-medium ${item.active ? 'text-primary-900' : 'text-gray-600'}`}>
                        {item.title}
                      </p>
                      <p className={`text-sm ${item.active ? 'text-primary-600' : 'text-gray-500'}`}>
                        {item.description}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-amber-50 border border-amber-200 rounded-2xl p-6">
              <div className="flex items-start space-x-3">
                <AlertCircle className="h-6 w-6 text-amber-600 mt-0.5 flex-shrink-0" />
                <div>
                  <h3 className="text-lg font-semibold text-amber-900 mb-2">Important Notes</h3>
                  <ul className="space-y-2 text-sm text-amber-700">
                    <li>• Ensure good image quality with proper lighting</li>
                    <li>• This tool is for screening purposes only</li>
                    <li>• Consult with a healthcare professional for diagnosis</li>
                    <li>• Your images are processed securely and privately</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-2xl shadow-sm border border-primary-100 p-6">
              <h3 className="text-lg font-semibold text-primary-900 mb-4">AI Models Used</h3>
              <div className="grid grid-cols-2 gap-3">
                {['ResNet50', 'EfficientNet', 'VGG16', 'MobileNetV4'].map((model) => (
                  <div key={model} className="bg-primary-50 rounded-lg p-3 text-center">
                    <p className="text-sm font-medium text-primary-800">{model}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Predict;
