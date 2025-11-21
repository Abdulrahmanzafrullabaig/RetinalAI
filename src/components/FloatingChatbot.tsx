import React, { useState, useRef, useEffect } from 'react';
import { MessageCircle, X, Send, Bot, User, Minimize2, Maximize2, Sparkles, Heart, Shield } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

interface Message {
  id: string;
  text: string;
  isBot: boolean;
  timestamp: Date;
  suggestions?: string[];
  type?: 'text' | 'formatted';
  formattedContent?: FormattedContent;
}

interface FormattedContent {
  title?: string;
  sections?: Array<{
    heading: string;
    content: string[];
    type?: 'list' | 'numbered' | 'text' | 'highlight';
    icon?: string;
  }>;
  highlights?: string[];
  callToAction?: string;
  severity?: 'info' | 'warning' | 'success' | 'urgent';
}

const FloatingChatbot: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: "🩺 Welcome to RetinalAI Medical Assistant!",
      isBot: true,
      timestamp: new Date(),
      type: 'formatted',
      formattedContent: {
        title: "🏥 Your Comprehensive Eye Health Guide",
        sections: [
          {
            heading: " Our Advanced AI System",
            icon: "🤖",
            content: [
              "95%+ diagnostic accuracy using ensemble models",
              "Real-time retinal image analysis",
              "Grad-CAM explainability for transparent results",
              "Multi-architecture deep learning (ResNet, EfficientNet, VGG, MobileNet)"
            ],
            type: "list"
          },
          {
            heading: " Diabetic Retinopathy Information",
            icon: "👁️",
            content: [
              "Complete guide to all 5 disease stages (0-4)",
              "Symptoms, causes, and risk factors",
              "Early detection and screening guidelines",
              "Latest treatment and management options"
            ],
            type: "list"
          },
          {
            heading: " Prevention & Care",
            icon: "🛡️",
            content: [
              "Evidence-based prevention strategies",
              "Lifestyle modifications for eye health",
              "Blood sugar and pressure management",
              "Personalized recommendations"
            ],
            type: "list"
          }
        ],
        highlights: [
          "🎯 Personalized medical guidance",
          "📊 Detailed stage explanations",
          "💊 Treatment recommendations",
          "📱 Easy appointment booking"
        ],
        callToAction: "What specific information can I help you with today?",
        severity: "info"
      },
      suggestions: [
        "🔬 How does RetinalAI diagnosis work?",
        "👁️ What is diabetic retinopathy?",
        "📊 Explain the disease stages",
        "🛡️ Prevention and care tips",
        "💊 Treatment options available"
      ]
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { user } = useAuth();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async (text: string) => {
    if (!text.trim()) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      text: text.trim(),
      isBot: false,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsTyping(true);

    try {
      const response = await fetch('/api/chatbot', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: text.trim(),
          isAuthenticated: !!user
        })
      });

      const data = await response.json();

      if (data.success) {
        const botMessage: Message = {
          id: (Date.now() + 1).toString(),
          text: data.response,
          isBot: true,
          timestamp: new Date(),
          suggestions: data.suggestions || [],
          type: data.formattedContent ? 'formatted' : 'text',
          formattedContent: data.formattedContent
        };
        setMessages(prev => [...prev, botMessage]);
      } else {
        throw new Error(data.message || 'Failed to get response');
      }
    } catch (error) {
      console.error('Chatbot error:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: "I'm sorry, I'm having trouble connecting right now. Please try again in a moment.",
        isBot: true,
        timestamp: new Date(),
        suggestions: ["Try again", "Contact support"]
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage(inputText);
  };

  const handleSuggestionClick = (suggestion: string) => {
    sendMessage(suggestion);
  };

  const formatMessage = (text: string) => {
    // Clean up markdown symbols and convert to HTML
    return text
      .replace(/#{1,6}\s*/g, '') // Remove heading markdown (###, ##, etc.)
      .replace(/\*\*\*(.*?)\*\*\*/g, '<strong><em>$1</em></strong>') // Bold + italic
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold
      .replace(/\*(.*?)\*/g, '<em>$1</em>') // Italic
      .replace(/\n/g, '<br>') // Line breaks
      .replace(/• /g, '• ') // Bullet points
      .replace(/^\s*[-*+]\s+/gm, '• '); // Convert list markers to bullet points
  };

  const renderFormattedContent = (content: FormattedContent) => {
    const getSeverityColors = (severity?: string) => {
      switch (severity) {
        case 'urgent': return 'bg-red-50 border-red-200 text-red-800';
        case 'warning': return 'bg-yellow-50 border-yellow-200 text-yellow-800';
        case 'success': return 'bg-green-50 border-green-200 text-green-800';
        default: return 'bg-primary-50 border-primary-200 text-primary-800';
      }
    };

    return (
      <div className="space-y-4">
        {content.title && (
          <div className={`p-3 rounded-lg border ${getSeverityColors(content.severity)}`}>
            <h4 className="font-heading font-bold text-sm mb-1">{content.title}</h4>
          </div>
        )}
        
        {content.sections?.map((section, index) => (
          <div key={index} className="border-l-4 border-primary-300 pl-4 py-2">
            <h5 className="font-heading font-semibold text-sm text-primary-700 mb-2 flex items-center gap-1">
              {section.icon && <span>{section.icon}</span>}
              {section.heading}
            </h5>
            
            {section.type === 'numbered' ? (
              <ol className="space-y-1 ml-4">
                {section.content.map((item, idx) => (
                  <li key={idx} className="text-sm font-body text-gray-700 list-decimal">
                    {item}
                  </li>
                ))}
              </ol>
            ) : section.type === 'list' ? (
              <ul className="space-y-1 ml-4">
                {section.content.map((item, idx) => (
                  <li key={idx} className="text-sm font-body text-gray-700 list-disc">
                    {item}
                  </li>
                ))}
              </ul>
            ) : section.type === 'highlight' ? (
              <div className="bg-gradient-to-r from-primary-50 to-gold-50 p-3 rounded-lg">
                {section.content.map((item, idx) => (
                  <p key={idx} className="text-sm font-body text-gray-800 font-medium">
                    ✨ {item}
                  </p>
                ))}
              </div>
            ) : (
              <div className="space-y-1">
                {section.content.map((item, idx) => (
                  <p key={idx} className="text-sm font-body text-gray-700">
                    {item}
                  </p>
                ))}
              </div>
            )}
          </div>
        ))}
        
        {content.highlights && content.highlights.length > 0 && (
          <div className="bg-gradient-to-r from-primary-50 via-gold-50 to-background-50 p-4 rounded-lg border border-primary-200">
            <h6 className="font-heading font-semibold text-sm text-primary-700 mb-2 flex items-center gap-1">
              ✨ Key Features
            </h6>
            <div className="grid grid-cols-1 gap-2">
              {content.highlights.map((highlight, index) => (
                <div key={index} className="flex items-center gap-2 text-sm font-body text-gray-700">
                  <div className="w-2 h-2 bg-gold-400 rounded-full"></div>
                  {highlight}
                </div>
              ))}
            </div>
          </div>
        )}
        
        {content.callToAction && (
          <div className="bg-gradient-to-r from-primary-500 to-gold-500 text-white p-3 rounded-lg text-center">
            <p className="text-sm font-body font-medium">
              💬 {content.callToAction}
            </p>
          </div>
        )}
      </div>
    );
  };

  if (!isOpen) {
    return (
      <div className="fixed bottom-6 right-6 z-50 group">
        <button
          onClick={() => setIsOpen(true)}
          className="relative bg-primary-500 hover:bg-primary-600 text-white rounded-full p-4 shadow-xl transition-all duration-300 hover:scale-110 transform hover:rotate-12"
          aria-label="Open medical assistant"
        >
          <Bot size={28} className="animate-pulse" />
          <div className="absolute -top-2 -right-2 w-6 h-6 bg-gold-500 rounded-full flex items-center justify-center animate-bounce">
            <Heart size={12} className="text-white" />
          </div>
          <Sparkles className="absolute top-1 right-1 w-4 h-4 text-yellow-300 animate-ping" />
        </button>
        
        <div className="absolute bottom-full right-0 mb-4 opacity-0 group-hover:opacity-100 transition-all duration-300 transform group-hover:translate-y-0 translate-y-2">
          <div className="bg-primary-800 text-white px-4 py-3 rounded-xl shadow-lg max-w-xs">
            <div className="flex items-center gap-2 mb-1">
              <Shield className="w-4 h-4 text-gold-400" />
              <span className="font-heading font-semibold text-sm">RetinalAI Assistant</span>
            </div>
            <p className="text-xs font-body text-primary-100">
              Get expert guidance on diabetic retinopathy, prevention tips, and treatment options!
            </p>
          </div>
          <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-8 border-r-8 border-t-8 border-transparent border-t-primary-800"></div>
        </div>
      </div>
    );
  }

  return (
    <div className={`fixed bottom-6 right-6 z-50 transition-all duration-300 ${
      isMinimized ? 'w-80 h-16' : 'w-96 h-[32rem]'
    }`}>
      <div className="bg-white rounded-lg shadow-2xl border border-primary-200 h-full flex flex-col overflow-hidden">
        {/* Header */}
        <div className="bg-primary-500 text-white p-4 flex items-center justify-between rounded-t-lg">
          <div className="flex items-center space-x-3">
            <div className="bg-white bg-opacity-20 rounded-full p-2 backdrop-blur-sm">
              <Bot size={24} className="animate-pulse" />
            </div>
            <div>
              <h3 className="font-heading font-bold text-sm flex items-center gap-1">
                RetinalAI Assistant
                <Sparkles size={14} className="text-yellow-300" />
              </h3>
              <p className="text-xs font-body text-primary-100 flex items-center gap-1">
                <Shield size={12} />
                Medical Information & AI Guidance
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setIsMinimized(!isMinimized)}
              className="text-white hover:bg-white hover:bg-opacity-20 rounded-full p-2 transition-all duration-200 transform hover:scale-110"
              aria-label={isMinimized ? "Maximize" : "Minimize"}
            >
              {isMinimized ? <Maximize2 size={18} /> : <Minimize2 size={18} />}
            </button>
            <button
              onClick={() => setIsOpen(false)}
              className="text-white hover:bg-white hover:bg-opacity-20 rounded-full p-2 transition-all duration-200 transform hover:scale-110"
              aria-label="Close chat"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {!isMinimized && (
          <>
            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-background-100">
              {messages.map((message) => (
                <div key={message.id} className={`flex ${message.isBot ? 'justify-start' : 'justify-end'}`}>
                  <div className={`max-w-[80%] rounded-lg p-3 ${
                    message.isBot
                      ? 'bg-white text-gray-800 shadow-sm border border-primary-200'
                      : 'bg-primary-600 text-white'
                  }`}>
                    <div className="flex items-start space-x-2">
                      {message.isBot && (
                        <div className="bg-primary-100 rounded-full p-1 mt-1">
                          <Bot size={14} className="text-primary-600" />
                        </div>
                      )}
                      <div className="flex-1">
                        {message.type === 'formatted' && message.formattedContent ? (
                          renderFormattedContent(message.formattedContent)
                        ) : (
                          <div 
                            className="text-sm font-body leading-relaxed"
                            dangerouslySetInnerHTML={{ __html: formatMessage(message.text) }}
                          />
                        )}
                        {message.isBot && message.suggestions && message.suggestions.length > 0 && (
                          <div className="mt-4 space-y-2">
                            <p className="text-xs font-heading font-semibold text-primary-600 mb-3 flex items-center gap-1">
                              💬 Quick Actions:
                            </p>
                            <div className="grid grid-cols-1 gap-2">
                              {message.suggestions.map((suggestion, index) => (
                                <button
                                  key={index}
                                  onClick={() => handleSuggestionClick(suggestion)}
                                  className="text-left text-sm font-body bg-gradient-to-r from-primary-50 to-gold-50 hover:from-primary-100 hover:to-gold-100 text-primary-700 px-3 py-2 rounded-lg border border-primary-200 hover:border-primary-300 transition-all duration-200 transform hover:scale-[1.02] hover:shadow-sm"
                                >
                                  {suggestion}
                                </button>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                      {!message.isBot && (
                        <div className="bg-primary-800 rounded-full p-1 mt-1">
                          <User size={14} className="text-white" />
                        </div>
                      )}
                    </div>
                    <div className={`text-xs font-body mt-2 opacity-70 ${
                      message.isBot ? 'text-gray-500' : 'text-primary-100'
                    }`}>
                      {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </div>
                  </div>
                </div>
              ))}
              
              {isTyping && (
                <div className="flex justify-start">
                  <div className="bg-white rounded-lg p-3 shadow-sm border border-primary-200 max-w-[80%]">
                    <div className="flex items-center space-x-2">
                      <div className="bg-primary-100 rounded-full p-1">
                        <Bot size={14} className="text-primary-600" />
                      </div>
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-primary-400 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-primary-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                        <div className="w-2 h-2 bg-primary-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="border-t border-primary-200 p-4 bg-white">
              <form onSubmit={handleSubmit} className="flex space-x-2">
                <input
                  type="text"
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="Ask about symptoms, stages, prevention, treatment options..."
                  className="flex-1 border border-primary-300 rounded-lg px-4 py-3 text-sm font-body focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-background-50 focus:bg-white transition-all duration-200"
                  disabled={isTyping}
                />
                <button
                  type="submit"
                  disabled={!inputText.trim() || isTyping}
                  className="bg-gold-500 hover:bg-gold-600 disabled:bg-gray-400 text-white rounded-lg px-4 py-2 transition-colors"
                  aria-label="Send message"
                >
                  <Send size={16} />
                </button>
              </form>
              <p className="text-xs font-body text-gray-500 mt-3 text-center flex items-center justify-center gap-1">
                <Shield size={12} className="text-primary-500" />
                Ask about symptoms, stages, prevention, AI diagnosis, or treatment options
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default FloatingChatbot;