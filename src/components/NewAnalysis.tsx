import React from 'react';
import { useNavigate } from 'react-router-dom';
import Header from './Header';

const NewAnalysis: React.FC = () => {
    const navigate = useNavigate();

    const handleImageAnalysis = () => {
        navigate('/predict');
    };

    const handleClinicalAnalysis = () => {
        navigate('/clinical-predict');
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-background-100 via-background-50 to-primary-50">
            <Header />
            <div className="py-8 px-4">
                <div className="max-w-4xl mx-auto">
                {/* Header */}
                <div className="text-center mb-12">
                    <h1 className="text-4xl font-bold text-primary-900 mb-4">
                        Choose Analysis Type
                    </h1>
                    <p className="text-xl text-primary-700 max-w-2xl mx-auto">
                        Select your preferred method for diabetic retinopathy assessment
                    </p>
                </div>

                {/* Analysis Options */}
                <div className="grid md:grid-cols-2 gap-8 max-w-5xl mx-auto">
                    {/* Image Analysis Card */}
                    <div className="group bg-white rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-2 border border-primary-100">
                        <div className="p-8">
                            {/* Icon */}
                            <div className="mb-6 flex justify-center">
                                <div className="w-20 h-20 bg-gradient-to-br from-primary-500 to-primary-600 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                                    <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                    </svg>
                                </div>
                            </div>

                            {/* Content */}
                            <div className="text-center mb-8">
                                <h3 className="text-2xl font-bold text-primary-900 mb-4">
                                    Fundus Image Analysis
                                </h3>
                                <p className="text-primary-700 mb-6 leading-relaxed">
                                    Upload retinal fundus images for AI-powered analysis using our advanced deep learning models
                                </p>

                                {/* Features */}
                                <div className="space-y-3 mb-8">
                                    <div className="flex items-center justify-center text-sm text-primary-700">
                                        <svg className="w-4 h-4 text-gold-500 mr-2" fill="currentColor" viewBox="0 0 20 20">
                                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                        </svg>
                                        Multi-model ensemble analysis
                                    </div>
                                    <div className="flex items-center justify-center text-sm text-primary-700">
                                        <svg className="w-4 h-4 text-gold-500 mr-2" fill="currentColor" viewBox="0 0 20 20">
                                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                        </svg>
                                        Grad-CAM explanations
                                    </div>
                                    <div className="flex items-center justify-center text-sm text-primary-700">
                                        <svg className="w-4 h-4 text-gold-500 mr-2" fill="currentColor" viewBox="0 0 20 20">
                                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                        </svg>
                                        Detailed visual analysis
                                    </div>
                                </div>
                            </div>

                            {/* Button */}
                            <button
                                onClick={handleImageAnalysis}
                                className="w-full bg-gold-500 hover:bg-gold-600 text-white font-semibold py-4 px-6 rounded-xl transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl"
                            >
                                Upload Fundus Image
                            </button>
                        </div>
                    </div>

                    {/* Clinical Data Analysis Card */}
                    <div className="group bg-white rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-2 border border-primary-100">
                        <div className="p-8">
                            {/* Icon */}
                            <div className="mb-6 flex justify-center">
                                <div className="w-20 h-20 bg-gradient-to-br from-primary-500 to-primary-700 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                                    <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                    </svg>
                                </div>
                            </div>

                            {/* Content */}
                            <div className="text-center mb-8">
                                <h3 className="text-2xl font-bold text-primary-900 mb-4">
                                    Clinical Data Analysis
                                </h3>
                                <p className="text-primary-700 mb-6 leading-relaxed">
                                    Enter clinical information for risk assessment based on patient health data and medical history
                                </p>

                                {/* Features */}
                                <div className="space-y-3 mb-8">
                                    <div className="flex items-center justify-center text-sm text-primary-700">
                                        <svg className="w-4 h-4 text-gold-500 mr-2" fill="currentColor" viewBox="0 0 20 20">
                                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                        </svg>
                                        Quick assessment form
                                    </div>
                                    <div className="flex items-center justify-center text-sm text-primary-700">
                                        <svg className="w-4 h-4 text-gold-500 mr-2" fill="currentColor" viewBox="0 0 20 20">
                                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                        </svg>
                                        Risk factor analysis
                                    </div>
                                    <div className="flex items-center justify-center text-sm text-primary-700">
                                        <svg className="w-4 h-4 text-gold-500 mr-2" fill="currentColor" viewBox="0 0 20 20">
                                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                        </svg>
                                        Personalized recommendations
                                    </div>
                                </div>
                            </div>

                            {/* Button */}
                            <button
                                onClick={handleClinicalAnalysis}
                                className="w-full bg-gold-500 hover:bg-gold-600 text-white font-semibold py-4 px-6 rounded-xl transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl"
                            >
                                Enter Clinical Data
                            </button>
                        </div>
                    </div>
                </div>

                {/* Additional Info */}
                <div className="mt-12 text-center">
                    <div className="bg-white rounded-xl shadow-md p-6 max-w-3xl mx-auto border border-primary-100">
                        <h4 className="text-lg font-semibold text-primary-900 mb-3">
                            Which analysis should I choose?
                        </h4>
                        <div className="grid md:grid-cols-2 gap-6 text-sm text-primary-700">
                            <div>
                                <h5 className="font-semibold text-gold-600 mb-2">Choose Image Analysis if:</h5>
                                <ul className="space-y-1 text-left">
                                    <li>• You have retinal fundus images</li>
                                    <li>• You want detailed visual analysis</li>
                                    <li>• You need AI explanations with Grad-CAM</li>
                                </ul>
                            </div>
                            <div>
                                <h5 className="font-semibold text-gold-600 mb-2">Choose Clinical Analysis if:</h5>
                                <ul className="space-y-1 text-left">
                                    <li>• You have patient health data</li>
                                    <li>• You want quick risk assessment</li>
                                    <li>• Images are not available</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
                </div>
            </div>
        </div>
    );
};

export default NewAnalysis;