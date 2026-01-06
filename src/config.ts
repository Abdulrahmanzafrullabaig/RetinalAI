
const getApiUrl = () => {
    // In production, this env var should be set. 
    // If not set, it defaults to a likely backend URL or empty string (relative path)
    return import.meta.env.VITE_API_URL || 'http://localhost:5000';
};

export const API_URL = getApiUrl();
