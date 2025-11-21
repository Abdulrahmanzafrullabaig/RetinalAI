import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Header from './Header';

interface ClinicalData {
    Age: string;
    HbA1c: string;
    Gender: string;
    Stage: string;
    Blood_Pressure: string;
    Comorbidities: string[];
}

interface PredictionResult {
    prediction: string;
    confidence: string;
    probabilities: {
        negative: string;
        positive: string;
    };
    risk_level: string;
    date: string;
    report_id?: number;
}

const ClinicalPredict: React.FC = () => {
    const navigate = useNavigate();
    const [formData, setFormData] = useState<ClinicalData>({
        Age: '',
        HbA1c: '',
        Gender: '',
        Stage: '',
        Blood_Pressure: '',
        Comorbidities: []
    });
    
    const [isLoading, setIsLoading] = useState(false);
    const [result, setResult] = useState<PredictionResult | null>(null);
    const [error, setError] = useState<string>('');
    const [showRecommendations, setShowRecommendations] = useState(false);
    const [recommendations, setRecommendations] = useState<string>('');
    const [loadingRecommendations, setLoadingRecommendations] = useState(false);

    const comorbidityOptions = [
        'Diabetes', 'Liver Problem', 'Kidney Problem', 'Nerves Problem',
        'Heart Problem', 'Feet Problem', 'Skin Problem', 'Smoking'
    ];

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handleComorbidityChange = (comorbidity: string) => {
        setFormData(prev => ({
            ...prev,
            Comorbidities: prev.Comorbidities.includes(comorbidity)
                ? prev.Comorbidities.filter(c => c !== comorbidity)
                : [...prev.Comorbidities, comorbidity]
        }));
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);
        setError('');
        setResult(null);
        setShowRecommendations(false);

        try {
            // Convert comorbidities from underscore format to space format for backend
            const processedData = {
                ...formData,
                Comorbidities: formData.Comorbidities.map(c => c.replace('_', ' '))
            };

            console.log('Sending clinical data:', processedData);

            const response = await fetch('/api/predict-clinical', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                credentials: 'include',
                body: JSON.stringify(processedData)
            });

            const data = await response.json();
            console.log('Response data:', data);

            if (data.success) {
                setResult(data.result);
            } else {
                setError(data.message || 'Failed to get prediction');
            }
        } catch (err) {
            setError('Network error. Please try again.');
            console.error('Prediction error:', err);
        } finally {
            setIsLoading(false);
        }
    };

    const generateRecommendations = async () => {
        setLoadingRecommendations(true);
        setRecommendations('');

        try {
            // Simulate AI recommendation generation (you can integrate with real AI service)
            const riskFactors = formData.Comorbidities.join(', ') || 'None reported';
            const mockRecommendations = `Based on your clinical profile (Age: ${formData.Age}, HbA1c: ${formData.HbA1c}%, DR Stage: ${formData.Stage}, BP: ${formData.Blood_Pressure}, Risk factors: ${riskFactors}), here are personalized recommendations:

1. **Blood Sugar Management**: Your HbA1c of ${formData.HbA1c}% indicates the need for improved glucose control. Aim for HbA1c below 7% through medication adherence and dietary modifications.

2. **Blood Pressure Control**: With ${formData.Blood_Pressure}, blood pressure management is crucial. Follow DASH diet principles and consider medication adjustment with your physician.

3. **Regular Monitoring**: Given your current DR stage (${formData.Stage}), schedule eye exams every 3-6 months with an ophthalmologist for early detection of progression.

These recommendations are based on current medical guidelines and should be discussed with your healthcare provider for personalized treatment planning.`;

            // Simulate API delay
            await new Promise(resolve => setTimeout(resolve, 2000));
            setRecommendations(mockRecommendations);
        } catch (err) {
            setRecommendations('Error generating recommendations. Please try again.');
        } finally {
            setLoadingRecommendations(false);
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-background-100 via-background-50 to-primary-50">
            <Header />
            <div className="py-8 px-4">
                <div className="max-w-4xl mx-auto">
                    {/* Header */}
                    <div className="mb-8">
                        <button
                            onClick={() => navigate('/new-analysis')}
                            className="flex items-center text-primary-600 hover:text-primary-800 font-medium mb-4 transition-colors"
                        >
                            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                            </svg>
                            Back to Analysis Options
                        </button>
                        
                        <div className="bg-white rounded-2xl shadow-lg p-8 border border-primary-100">
                            <h1 className="text-3xl font-bold text-primary-700 text-center mb-4">
                                Clinical Data Risk Assessment
                            </h1>
                            <p className="text-primary-600 text-center max-w-2xl mx-auto">
                                Enter your clinical information for AI-powered diabetic retinopathy risk assessment
                            </p>
                        </div>
                    </div>

                {/* Prediction Form */}
                <div className="bg-white rounded-2xl shadow-lg border border-primary-100 p-8 mb-8">
                    <form onSubmit={handleSubmit} className="space-y-6">
                        {/* Numerical Inputs */}
                        <div className="grid sm:grid-cols-2 gap-6">
                            <div>
                                <label htmlFor="Age" className="block text-sm font-medium text-primary-700 mb-2">
                                    Age (Years):
                                </label>
                                <input
                                    type="number"
                                    id="Age"
                                    name="Age"
                                    required
                                    min="18"
                                    max="120"
                                    value={formData.Age}
                                    onChange={handleInputChange}
                                    className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-all"
                                />
                            </div>
                            <div>
                                <label htmlFor="HbA1c" className="block text-sm font-medium text-primary-700 mb-2">
                                    HbA1c (%):
                                </label>
                                <input
                                    type="number"
                                    id="HbA1c"
                                    name="HbA1c"
                                    step="0.01"
                                    required
                                    min="4.0"
                                    max="15.0"
                                    value={formData.HbA1c}
                                    onChange={handleInputChange}
                                    className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-all"
                                />
                            </div>
                        </div>

                        {/* Categorical Selects */}
                        <div className="grid sm:grid-cols-3 gap-6">
                            <div>
                                <label htmlFor="Gender" className="block text-sm font-medium text-primary-700 mb-2">
                                    Gender:
                                </label>
                                <select
                                    id="Gender"
                                    name="Gender"
                                    required
                                    value={formData.Gender}
                                    onChange={handleInputChange}
                                    className="w-full p-3 border border-gray-300 rounded-lg appearance-none bg-white focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-all"
                                >
                                    <option value="">Select...</option>
                                    <option value="Male">Male</option>
                                    <option value="Female">Female</option>
                                </select>
                            </div>
                            <div>
                                <label htmlFor="Stage" className="block text-sm font-medium text-primary-700 mb-2">
                                    DR Stage (Current/Last Known):
                                </label>
                                <select
                                    id="Stage"
                                    name="Stage"
                                    required
                                    value={formData.Stage}
                                    onChange={handleInputChange}
                                    className="w-full p-3 border border-gray-300 rounded-lg appearance-none bg-white focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-all"
                                >
                                    <option value="">Select...</option>
                                    <option value="None">None</option>
                                    <option value="Mild">Mild</option>
                                    <option value="Moderate">Moderate</option>
                                    <option value="Severe">Severe</option>
                                    <option value="Proliferative">Proliferative</option>
                                </select>
                            </div>
                            <div>
                                <label htmlFor="Blood_Pressure" className="block text-sm font-medium text-primary-700 mb-2">
                                    Blood Pressure Category:
                                </label>
                                <select
                                    id="Blood_Pressure"
                                    name="Blood_Pressure"
                                    required
                                    value={formData.Blood_Pressure}
                                    onChange={handleInputChange}
                                    className="w-full p-3 border border-gray-300 rounded-lg appearance-none bg-white focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-all"
                                >
                                    <option value="">Select...</option>
                                    <option value="Normal">Normal</option>
                                    <option value="Prehypertension">Prehypertension</option>
                                    <option value="Hypertension Stage 1">Hypertension Stage 1</option>
                                    <option value="Hypertension Stage 2">Hypertension Stage 2</option>
                                </select>
                            </div>
                        </div>

                        {/* Comorbidities Checkboxes */}
                        <div className="pt-4 border-t border-gray-200">
                            <p className="text-lg font-semibold text-primary-800 mb-4">
                                Comorbidities and Risk Factors:
                            </p>
                            <div className="grid sm:grid-cols-3 gap-4">
                                {comorbidityOptions.map((comorbidity) => (
                                    <label key={comorbidity} className="flex items-center space-x-2 text-primary-700 text-sm">
                                        <input
                                            type="checkbox"
                                            checked={formData.Comorbidities.includes(comorbidity)}
                                            onChange={() => handleComorbidityChange(comorbidity)}
                                            className="rounded text-primary-600 focus:ring-primary-500 h-4 w-4"
                                        />
                                        <span>{comorbidity}</span>
                                    </label>
                                ))}
                            </div>
                        </div>

                        {/* Submit Button */}
                        <button
                            type="submit"
                            disabled={isLoading}
                            className="w-full py-4 bg-gold-500 hover:bg-gold-600 disabled:bg-gray-400 text-white font-semibold rounded-lg shadow-md hover:shadow-lg focus:outline-none focus:ring-4 focus:ring-gold-500 focus:ring-opacity-50 transition-all transform hover:scale-105 disabled:hover:scale-100"
                        >
                            {isLoading ? (
                                <div className="flex items-center justify-center">
                                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                    Analyzing Clinical Data...
                                </div>
                            ) : (
                                'Get Risk Assessment'
                            )}
                        </button>
                    </form>
                </div>

                {/* Error Display */}
                {error && (
                    <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-8">
                        <div className="flex">
                            <svg className="h-5 w-5 text-red-400 mr-2" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                            </svg>
                            <p className="text-red-800">{error}</p>
                        </div>
                    </div>
                )}

                {/* Results Display */}
                {result && (
                    <div className="bg-white rounded-2xl shadow-lg border border-primary-100 p-8">
                        <h2 className="text-2xl font-bold text-primary-900 mb-6 text-center">
                            Risk Assessment Results
                        </h2>

                        {/* Main Result */}
                        <div className={`p-6 rounded-xl border-2 text-center mb-6 ${
                            result.risk_level === 'High' 
                            ? 'bg-red-50 border-red-200 text-red-800' 
                            : 'bg-green-50 border-green-200 text-green-800'
                        }`}>
                            <h3 className="text-xl font-bold mb-2">
                                {result.prediction}
                            </h3>
                            <p className="text-lg font-semibold">
                                Risk Level: {result.risk_level}
                            </p>
                            <p className="text-sm mt-2">
                                Confidence: {result.confidence}
                            </p>
                        </div>

                        {/* Probability Breakdown */}
                        <div className="grid md:grid-cols-2 gap-4 mb-6">
                            <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                                <h4 className="font-semibold text-green-800">Negative Probability</h4>
                                <p className="text-2xl font-bold text-green-600">{result.probabilities.negative}</p>
                            </div>
                            <div className="bg-red-50 p-4 rounded-lg border border-red-200">
                                <h4 className="font-semibold text-red-800">Positive Probability</h4>
                                <p className="text-2xl font-bold text-red-600">{result.probabilities.positive}</p>
                            </div>
                        </div>

                        {/* Generate Recommendations Button */}
                        <div className="text-center mb-6">
                            <button
                                onClick={() => {
                                    setShowRecommendations(true);
                                    generateRecommendations();
                                }}
                                disabled={loadingRecommendations}
                                className="py-3 px-6 bg-gold-500 hover:bg-gold-600 disabled:bg-gold-400 text-white font-medium rounded-lg shadow-md transition duration-150 ease-in-out transform hover:scale-105 disabled:hover:scale-100"
                            >
                                {loadingRecommendations ? (
                                    <div className="flex items-center">
                                        <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                        </svg>
                                        Generating Analysis...
                                    </div>
                                ) : (
                                    '✨ Generate Personalized Analysis'
                                )}
                            </button>
                        </div>

                        {/* Recommendations Display */}
                        {showRecommendations && recommendations && (
                            <div className="border-t pt-6">
                                <h3 className="text-xl font-semibold text-primary-800 mb-4">
                                    Personalized Recommendations
                                </h3>
                                <div className="bg-primary-50 p-6 rounded-lg border border-primary-200">
                                    <div className="whitespace-pre-line text-primary-700 leading-relaxed">
                                        {recommendations}
                                    </div>
                                </div>
                                <p className="text-sm mt-4 italic text-red-600 border-t pt-3">
                                    <strong className="font-bold">Medical Disclaimer:</strong> This AI-generated content is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified health provider.
                                </p>
                            </div>
                        )}

                        {/* Action Buttons */}
                        <div className="mt-6 flex gap-4 justify-center">
                            <button
                                onClick={() => navigate('/results')}
                                className="px-6 py-2 bg-primary-600 hover:bg-primary-700 text-white font-medium rounded-lg transition-colors"
                            >
                                View All Results
                            </button>
                            <button
                                onClick={() => {
                                    setResult(null);
                                    setShowRecommendations(false);
                                    setRecommendations('');
                                }}
                                className="px-6 py-2 bg-gray-600 hover:bg-gray-700 text-white font-medium rounded-lg transition-colors"
                            >
                                New Analysis
                            </button>
                        </div>
                    </div>
                )}
                </div>
            </div>
        </div>
    );
};

export default ClinicalPredict;