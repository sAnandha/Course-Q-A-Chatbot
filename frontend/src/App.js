import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import SourceViewer from './components/SourceViewer';
import './App.css';

// Configure axios base URL
axios.defaults.baseURL = process.env.NODE_ENV === 'production' ? '/api' : 'http://localhost:8000';

function App() {
  const [query, setQuery] = useState('');
  const [language, setLanguage] = useState('en');
  const [messages, setMessages] = useState([]);
  const [citations, setCitations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [selectedSource, setSelectedSource] = useState(null);
  const [questionId, setQuestionId] = useState(0);
  const [sessionId, setSessionId] = useState(null);
  const [sessionInfo, setSessionInfo] = useState(null);
  const [uploadedDocuments, setUploadedDocuments] = useState([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const [showDocuments, setShowDocuments] = useState(false);
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);
  const dropZoneRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Add global functions
  useEffect(() => {
    window.copyCode = (button) => {
      const codeBlock = button.nextElementSibling.querySelector('code');
      const text = codeBlock.textContent;
      navigator.clipboard.writeText(text).then(() => {
        button.textContent = '✅ Copied!';
        setTimeout(() => {
          button.textContent = '📋 Copy';
        }, 2000);
      });
    };
    
    window.handleCitationClick = (sourceId) => {
      handleSourceClick(sourceId);
    };
  }, []);

  useEffect(() => {
    // Create session on component mount
    createSession();
    // Load documents on mount and refresh
    loadUploadedDocuments();
  }, []);

  useEffect(() => {
    if (sessionId) {
      loadUploadedDocuments();
    }
  }, [sessionId]);

  const createSession = async () => {
    try {
      const response = await axios.post('/session/create');
      setSessionId(response.data.session_id);
      console.log('Session created:', response.data.session_id);
    } catch (err) {
      console.error('Failed to create session:', err);
    }
  };

  const getSessionInfo = async () => {
    if (!sessionId) return;
    try {
      const response = await axios.get(`/session/${sessionId}/info`);
      setSessionInfo(response.data);
    } catch (err) {
      console.error('Failed to get session info:', err);
    }
  };

  const loadUploadedDocuments = async () => {
    if (!sessionId) return;
    try {
      const headers = { 'X-Session-ID': sessionId };
      const response = await axios.get('/debug/documents', { headers });
      setUploadedDocuments(response.data.documents || []);
    } catch (err) {
      console.error('Failed to load documents:', err);
    }
  };

  const deleteSession = async () => {
    if (!sessionId) return;
    try {
      await axios.delete(`/session/${sessionId}`);
      // Clear all local state
      setMessages([]);
      setCitations([]);
      setUploadedDocuments([]);
      setSessionInfo(null);
      setSelectedSource(null);
      setError('');
      
      // Create new session
      await createSession();
      
      const systemMessage = {
        type: 'system',
        content: 'Session cleared! All documents and chat history have been reset.'
      };
      setMessages([systemMessage]);
    } catch (err) {
      console.error('Failed to delete session:', err);
      setError('Failed to clear session');
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    if (!dropZoneRef.current?.contains(e.relatedTarget)) {
      setIsDragOver(false);
    }
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    const validFiles = files.filter(file => 
      file.type === 'application/pdf' || 
      file.name.endsWith('.md') || 
      file.name.endsWith('.csv')
    );
    
    if (validFiles.length === 0) {
      setError('Please upload PDF, Markdown (.md), or CSV files only.');
      return;
    }
    
    for (const file of validFiles) {
      await uploadFile(file);
    }
  };

  const uploadFile = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      setLoading(true);
      const headers = { 'Content-Type': 'multipart/form-data' };
      if (sessionId) headers['X-Session-ID'] = sessionId;
      
      const response = await axios.post('/upload', formData, { headers });
      
      const uploadMessage = {
        type: 'system',
        content: `Successfully uploaded "${file.name}"`
      };
      setMessages(prev => [...prev, uploadMessage]);
      
      // Refresh documents and session info
      setTimeout(() => {
        loadUploadedDocuments();
        getSessionInfo();
      }, 500);
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  const formatAnswer = (text) => {
    return text
      .replace(/### (.*?)\n/g, '<h3>$1</h3>')
      .replace(/## (.*?)\n/g, '<h2>$1</h2>')
      .replace(/```([\s\S]*?)```/g, '<div class="code-block"><button class="copy-btn" onclick="copyCode(this)">📋 Copy</button><pre><code>$1</code></pre></div>')
      .replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>')
      .replace(/• (.*?)\n/g, '<li>$1</li>')
      .replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>')
      .replace(/\[S(\d+):([^:]+):pp(\d+)\]/g, '<span class="citation" onclick="handleCitationClick(\'S$1:$2:pp$3\')">[S$1:$2:pp$3]</span>')
      .replace(/\[S(\d+):pp(\d+)\]/g, '<span class="citation" onclick="handleCitationClick(\'S$1:pp$2\')">[S$1:pp$2]</span>')
      .replace(/\[S(\d+)\]/g, '<span class="citation" onclick="handleCitationClick(\'S$1\')">[S$1]</span>')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/^#+\s*$/gm, '')  // Remove standalone # symbols
      .replace(/\n\n/g, '<br><br>');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim() || loading) return;

    const userMessage = { type: 'user', content: query };
    setMessages(prev => [...prev, userMessage]);
    
    setLoading(true);
    setError('');
    setSelectedSource(null);
    setQuestionId(prev => prev + 1);
    
    const currentQuery = query;
    setQuery('');

    try {
      const headers = sessionId ? { 'X-Session-ID': sessionId } : {};
      const response = await axios.post('/answer', {
        query: currentQuery,
        lang: language,
        top_k: 5
      }, { headers });

      const assistantMessage = {
        type: 'assistant',
        content: response.data.answer,
        citations: response.data.citations || []
      };

      setMessages(prev => [...prev, assistantMessage]);
      setCitations(response.data.citations || []);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred');
      const errorMessage = {
        type: 'assistant',
        content: 'Sorry, I encountered an error processing your question.',
        error: true
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleSourceClick = async (sourceId) => {
    try {
      const headers = sessionId ? { 'X-Session-ID': sessionId } : {};
      const response = await axios.get(`/source/${sourceId}`, { headers });
      setSelectedSource({
        id: sourceId,
        text: response.data.text,
        metadata: response.data.metadata,
        page_number: response.data.page_number,
        document_name: response.data.document_name
      });
    } catch (err) {
      console.error('Error fetching source:', err);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    await uploadFile(file);
    // Reset file input
    e.target.value = '';
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const downloadSessionPDF = async () => {
    if (!sessionId) return;
    
    try {
      const response = await axios.get(`/session/${sessionId}/export`, {
        responseType: 'blob'
      });
      
      const blob = new Blob([response.data], { type: 'application/pdf' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `session_${sessionId.substring(0, 8)}_qa.pdf`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Failed to download PDF:', err);
      if (err.response?.status === 400) {
        setError('PDF export is only available for English questions. Hindi PDF conversion is under development.');
      } else {
        setError('Failed to download session PDF');
      }
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-left">
          <h1>Course Q&A Assistant</h1>
          {sessionId && (
            <span className="session-badge">Session: {sessionId.substring(0, 8)}...</span>
          )}
        </div>
        <div className="header-controls">
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.md,.csv"
            onChange={handleFileUpload}
            disabled={loading}
            multiple
          />
          <button 
            className="upload-button"
            onClick={() => fileInputRef.current?.click()}
            disabled={loading}
          >
            Upload Files
          </button>
          <button 
            className="toggle-docs-button"
            onClick={() => setShowDocuments(!showDocuments)}
          >
            Documents ({uploadedDocuments.length})
          </button>
          <button 
            className="download-button"
            onClick={() => downloadSessionPDF()}
            disabled={!sessionId}
            title="Export English Q&A to PDF (Hindi conversion under development)"
          >
            Download Q&A
          </button>
          <button 
            className="clear-button"
            onClick={deleteSession}
            disabled={!sessionId}
          >
            Clear Session
          </button>
        </div>
      </header>

      <div className="main-content">
        {/* Documents Panel */}
        {showDocuments && (
          <div className="documents-panel">
            <div className="documents-header">
              <h3>Uploaded Documents</h3>
              <button 
                className="close-panel-btn"
                onClick={() => setShowDocuments(false)}
              >
                ×
              </button>
            </div>
            

            
            {/* Documents List */}
            <div className="documents-list">
              {uploadedDocuments.length === 0 ? (
                <div className="no-documents">
                  <p>No documents uploaded yet</p>
                  <p className="hint">Upload some documents to get started!</p>
                </div>
              ) : (
                uploadedDocuments.map((doc, index) => (
                  <div key={index} className="document-item">
                    <div className="doc-icon">DOC</div>
                    <div className="doc-info">
                      <div className="doc-name">{doc.doc_id || `Document ${index + 1}`}</div>
                      <div className="doc-preview">{doc.text_preview}</div>
                      <div className="doc-meta">
                        Original Document
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
            

          </div>
        )}
        
        <div 
          ref={dropZoneRef}
          className={`chat-section ${isDragOver ? 'drag-over' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {isDragOver && (
            <div className="drop-overlay">
              <div className="drop-overlay-content">
                <div className="drop-icon">+</div>
                <p>Drop files here to upload</p>
                <div className="supported-formats">
                  <span>PDF</span> • <span>Markdown</span> • <span>CSV</span>
                </div>
              </div>
            </div>
          )}
          
          <div className="chat-messages">
            {messages.length === 0 && (
              <div className="welcome-message">
                <h2>Welcome to Your AI Course Assistant</h2>
                <p>I'm here to help you learn! Upload your course materials and ask me anything.</p>
                
                <div className="welcome-features">
                  <div className="feature">
                    <span className="feature-icon">FILES</span>
                    <span>Upload PDFs, Markdown, or CSV files</span>
                  </div>
                  <div className="feature">
                    <span className="feature-icon">SEARCH</span>
                    <span>Get answers with precise citations</span>
                  </div>
                  <div className="feature">
                    <span className="feature-icon">LANG</span>
                    <span>Ask questions in English or Hindi</span>
                  </div>
                  <div className="feature">
                    <span className="feature-icon">PDF</span>
                    <span>Download available for English conversations only</span>
                  </div>
                </div>
                
                {uploadedDocuments.length === 0 ? (
                  <div className="getting-started">
                    <p className="start-hint">Start by uploading some documents above!</p>
                  </div>
                ) : (
                  <div className="ready-to-chat">
                    <p className="ready-hint">{uploadedDocuments.length} document(s) ready. Ask me anything!</p>
                  </div>
                )}
              </div>
            )}
            
            {messages.map((message, index) => (
              <div key={index} className={`message ${message.type}`}>
                <div className="message-avatar">
                  {message.type === 'user' ? 'U' : message.type === 'system' ? 'S' : 'AI'}
                </div>
                <div className={`message-content ${message.error ? 'error' : ''} ${message.type === 'assistant' ? 'formatted' : ''}`}>
                  {message.type === 'assistant' ? (
                    <div dangerouslySetInnerHTML={{ __html: formatAnswer(message.content) }} />
                  ) : (
                    message.content
                  )}
                </div>
              </div>
            ))}
            
            {loading && (
              <div className="message assistant">
                <div className="message-avatar">AI</div>
                <div className="message-content">
                  <div className="loading-indicator">
                    Thinking
                    <div className="loading-dots">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="query-form">
            <div className="form-container">
              <div className="form-group">
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder={uploadedDocuments.length > 0 ? "Ask me anything about your documents..." : "Upload documents first, then ask questions..."}
                  disabled={loading || uploadedDocuments.length === 0}
                />
              </div>
              
              <div className="form-controls">
                <select
                  value={language}
                  onChange={(e) => setLanguage(e.target.value)}
                  disabled={loading}
                  className="language-select"
                >
                  <option value="en">English</option>
                  <option value="hi">हिंदी (Hindi)</option>
                </select>
                
                <button 
                  type="submit" 
                  disabled={loading || !query.trim() || uploadedDocuments.length === 0}
                  className="send-button"
                  title={uploadedDocuments.length === 0 ? "Upload documents first" : "Send message"}
                >
                  {loading ? '...' : '➤'}
                </button>
              </div>
            </div>
          </form>
        </div>

        <div className="sources-section">
          {selectedSource ? (
            <div className="source-detail">
              <div className="source-detail-header">
                <h3>Source Details</h3>
                <button 
                  onClick={() => setSelectedSource(null)}
                  className="back-button"
                >
                  ← Back
                </button>
              </div>
              <div className="source-detail-content">
                <h4>Source ID: {selectedSource.id}</h4>
                <p>Page {selectedSource.page_number || 'Unknown'} from document {selectedSource.document_name || 'Unknown'}</p>
                <div className="source-text">
                  {selectedSource.text}
                </div>
              </div>
            </div>
          ) : (
            <SourceViewer 
              citations={citations} 
              onSourceClick={handleSourceClick}
              key={questionId}
            />
          )}
        </div>
      </div>
    </div>
  );
}

export default App;