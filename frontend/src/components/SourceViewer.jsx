import React, { useState } from 'react';

const SourceViewer = ({ citations, onSourceClick }) => {
  const [expandedSources, setExpandedSources] = useState(new Set());

  const toggleSource = (sourceId) => {
    const newExpanded = new Set(expandedSources);
    if (newExpanded.has(sourceId)) {
      newExpanded.delete(sourceId);
    } else {
      newExpanded.add(sourceId);
    }
    setExpandedSources(newExpanded);
  };

  if (!citations || citations.length === 0) {
    return (
      <div className="source-viewer">
        <h3>Sources</h3>
        <div style={{ padding: '20px', textAlign: 'center', color: '#6b7280' }}>
          <p>No sources available</p>
          <p style={{ fontSize: '12px', marginTop: '8px' }}>
            Ask a question to see relevant sources
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="source-viewer">
      <h3>Sources ({citations.length})</h3>
      
      <div className="sources-list">
        {citations.map((citation, index) => {
          const isExpanded = expandedSources.has(citation.source_id);
          
          return (
            <div 
              key={citation.source_id} 
              className="source-card"
              onClick={() => onSourceClick && onSourceClick(citation.source_id)}
            >
              <div className="source-header">
                <div className="source-title">
                  {citation.document_name ? (
                    <>
                      <div className="doc-name">{citation.document_name}</div>
                      <div className="source-id">Source {index + 1}</div>
                    </>
                  ) : (
                    <div className="source-id">Source {index + 1}</div>
                  )}
                </div>
                
                <div className="source-badges">
                  {citation.page_number && (
                    <span className="page-badge">
                      p.{citation.page_number}
                    </span>
                  )}
                  <span className="confidence-badge">
                    {Math.round(citation.confidence * 100)}%
                  </span>
                  <button 
                    className="expand-button"
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleSource(citation.source_id);
                    }}
                  >
                    {isExpanded ? '−' : '+'}
                  </button>
                </div>
              </div>
              
              <div className={`source-content ${isExpanded ? '' : 'collapsed'}`}>
                {citation.span}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default SourceViewer;