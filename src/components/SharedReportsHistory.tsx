
import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Share, Calendar, User, FileText, CheckCircle, ArrowLeft, RefreshCcw } from 'lucide-react';
import Header from './Header';
import { API_URL } from '../config';

interface SharedReport {
    id: number;
    filename: string;
    result: string;
    report_date: string;
    doctor_name: string;
    shared_at: string;
    status: string;
    original_report_id: number;
}

const SharedReportsHistory = () => {
    const [sharedReports, setSharedReports] = useState<SharedReport[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        const fetchSharedReports = async () => {
            setLoading(true);
            setError('');
            try {
                const response = await fetch(`${API_URL}/api/shared-reports-history`, {
                    method: 'GET',
                    credentials: 'include',
                });
                if (response.ok) {
                    const data = await response.json();
                    setSharedReports(data);
                } else {
                    const errorData = await response.json();
                    setError(errorData.message || 'Failed to load shared reports history');
                }
            } catch (err) {
                console.error('Error fetching shared reports:', err);
                setError('Error fetching shared reports. Please check your connection.');
            } finally {
                setLoading(false);
            }
        };

        fetchSharedReports();
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

    if (loading) {
        return (
            <div className="min-h-screen bg-background-100 flex items-center justify-center">
                <div className="flex items-center space-x-2 text-primary-700">
                    <RefreshCcw className="h-5 w-5 animate-spin" />
                    <span>Loading History...</span>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-background-100">
            <Header />
            <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <div className="flex items-center justify-between mb-8">
                    <div className="flex items-center space-x-4">
                        <Link
                            to="/patient-dashboard"
                            className="p-2 text-primary-600 hover:text-primary-800 hover:bg-primary-100 rounded-lg transition-colors"
                            title="Back to Dashboard"
                        >
                            <ArrowLeft className="h-6 w-6" />
                        </Link>
                        <div>
                            <h1 className="text-3xl font-bold text-primary-900 mb-2">Shared Reports History</h1>
                            <p className="text-primary-600">Overview of reports shared with doctors</p>
                        </div>
                    </div>
                </div>

                {error && (
                    <div className="mb-6 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm" role="alert">
                        {error}
                    </div>
                )}

                <div className="bg-white rounded-2xl shadow-sm border border-primary-100 overflow-hidden">
                    {sharedReports.length > 0 ? (
                        <div className="overflow-x-auto">
                            <table className="w-full text-left border-collapse">
                                <thead>
                                    <tr className="bg-primary-50 border-b border-primary-100">
                                        <th className="px-6 py-4 text-sm font-semibold text-primary-900">Report Details</th>
                                        <th className="px-6 py-4 text-sm font-semibold text-primary-900">Diagnosis</th>
                                        <th className="px-6 py-4 text-sm font-semibold text-primary-900">Shared With</th>
                                        <th className="px-6 py-4 text-sm font-semibold text-primary-900">Shared Date</th>
                                        <th className="px-6 py-4 text-sm font-semibold text-primary-900">Status</th>
                                        <th className="px-6 py-4 text-sm font-semibold text-primary-900">Actions</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-gray-100">
                                    {sharedReports.map((report) => (
                                        <tr key={report.id} className="hover:bg-gray-50 transition-colors">
                                            <td className="px-6 py-4">
                                                <div className="flex items-center space-x-3">
                                                    <div className="bg-primary-100 p-2 rounded-lg">
                                                        <FileText className="h-5 w-5 text-primary-600" />
                                                    </div>
                                                    <div>
                                                        <p className="text-sm font-medium text-primary-900">Report #{report.original_report_id}</p>
                                                        <p className="text-xs text-primary-600">{new Date(report.report_date).toLocaleDateString()}</p>
                                                    </div>
                                                </div>
                                            </td>
                                            <td className="px-6 py-4">
                                                <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getResultColor(report.result)}`}>
                                                    {report.result}
                                                </span>
                                            </td>
                                            <td className="px-6 py-4">
                                                <div className="flex items-center space-x-2">
                                                    <User className="h-4 w-4 text-primary-400" />
                                                    <span className="text-sm text-primary-700 font-medium">{report.doctor_name}</span>
                                                </div>
                                            </td>
                                            <td className="px-6 py-4">
                                                <div className="flex items-center space-x-2 text-sm text-primary-600">
                                                    <Calendar className="h-4 w-4" />
                                                    <span>
                                                        {new Date(report.shared_at).toLocaleDateString()}
                                                        <span className="text-gray-400 ml-1">
                                                            {new Date(report.shared_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                                        </span>
                                                    </span>
                                                </div>
                                            </td>
                                            <td className="px-6 py-4">
                                                <div className="flex items-center space-x-1.5">
                                                    {report.status === 'reviewed' ?
                                                        <CheckCircle className="h-4 w-4 text-green-500" /> :
                                                        <div className="h-2 w-2 rounded-full bg-yellow-400"></div>
                                                    }
                                                    <span className={`text-sm font-medium capitalize ${report.status === 'reviewed' ? 'text-green-700' :
                                                            report.status === 'urgent' ? 'text-red-600' : 'text-yellow-700'
                                                        }`}>
                                                        {report.status}
                                                    </span>
                                                </div>
                                            </td>
                                            <td className="px-6 py-4">
                                                <Link
                                                    to={`/result/${report.original_report_id}`}
                                                    className="text-primary-600 hover:text-primary-800 font-medium text-sm hover:underline"
                                                >
                                                    View Report
                                                </Link>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    ) : (
                        <div className="text-center py-12">
                            <div className="bg-primary-50 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                                <Share className="h-8 w-8 text-primary-400" />
                            </div>
                            <h3 className="text-lg font-medium text-primary-900 mb-2">No Reports Shared Yet</h3>
                            <p className="text-primary-600 mb-6 max-w-sm mx-auto">
                                When you share your analysis results with a doctor, they will appear here.
                            </p>
                            <Link
                                to="/results"
                                className="inline-flex items-center space-x-2 bg-primary-600 hover:bg-primary-700 text-white px-6 py-3 rounded-lg font-medium transition-colors"
                            >
                                <FileText className="h-5 w-5" />
                                <span>Go to Results</span>
                            </Link>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default SharedReportsHistory;
