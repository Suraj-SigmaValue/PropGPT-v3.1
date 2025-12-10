import { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import Select from 'react-select';
import AsyncSelect from 'react-select/async';
import { streamQuery, getComparisonItems, submitFeedback } from './services/api';
import './index.css';

function App() {
  const [comparisonType, setComparisonType] = useState('Location');
  const [selectedItems, setSelectedItems] = useState([]);
  const [categories, setCategories] = useState(['General']);
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [streamStatus, setStreamStatus] = useState('');
  const [lastResponse, setLastResponse] = useState(null);

  const messagesEndRef = useRef(null);

  const comparisonTypes = [
    { value: 'Location', label: 'Location' },
    { value: 'City', label: 'City' },
    { value: 'Project', label: 'Project' }
  ];

  const categoryOptions = [
    { value: 'General', label: 'General' },
    { value: 'Demand', label: 'Demand' },
    { value: 'Supply', label: 'Supply' },
    { value: 'Pricing', label: 'Pricing' },
    { value: 'Demographics', label: 'Demographics' }
  ];

  const loadOptions = async (inputValue) => {
    try {
      const data = await getComparisonItems(comparisonType, inputValue, 50);
      return data.items.map(item => ({
        value: item,
        label: item.charAt(0).toUpperCase() + item.slice(1)
      }));
    } catch (error) {
      console.error('Error loading items:', error);
      return [];
    }
  };

  useEffect(() => {
    setSelectedItems([]);
  }, [comparisonType]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamStatus]);

  const handleSubmitQuery = async (e) => {
    e.preventDefault();
    if (!query.trim() || selectedItems.length === 0) return;

    const userMessage = { role: 'user', content: query };
    setMessages(prev => [...prev, userMessage]);

    const assistantMessageId = Date.now();
    setMessages(prev => [...prev, {
      id: assistantMessageId,
      role: 'assistant',
      content: '', // Start empty
      metadata: null,
      isStreaming: true
    }]);

    setLoading(true);
    setStreamStatus('Initializing...');
    setQuery('');

    const onChunk = (token) => {
      setMessages(prev => prev.map(msg => {
        if (msg.id === assistantMessageId) {
          return { ...msg, content: msg.content + token };
        }
        return msg;
      }));
    };

    const onStatus = (status) => {
      setStreamStatus(status);
    };

    const onMetadata = (metadata) => {
      setMessages(prev => prev.map(msg => {
        if (msg.id === assistantMessageId) {
          return { ...msg, metadata: metadata, isStreaming: false };
        }
        return msg;
      }));
      setLastResponse({
        query: userMessage.content,
        items: selectedItems.map(item => item.value),
        categories,
        mapping_keys: metadata.mapping_keys,
        comparison_type: comparisonType,
        provider: metadata.response_provider
      });
    };

    const onError = (errorMsg) => {
      setMessages(prev => prev.map(msg => {
        if (msg.id === assistantMessageId) {
          return {
            ...msg,
            content: msg.content + `\n\n**Error:** ${errorMsg}`,
            isError: true,
            isStreaming: false
          };
        }
        return msg;
      }));
      setLoading(false);
      setStreamStatus('');
    };

    await streamQuery(
      {
        query: query,
        comparison_type: comparisonType,
        items: selectedItems.map(item => item.value),
        categories: categories,
        mapping_llm_provider: 'openai',
        response_llm_provider: 'openai'
      },
      onChunk, onStatus, onMetadata, onError
    );

    setLoading(false);
    setStreamStatus('');
  };

  const handleFeedback = async (feedbackType) => {
    if (!lastResponse) return;
    try {
      await submitFeedback({
        feedback_type: feedbackType,
        ...lastResponse
      });
      setLastResponse(null);
    } catch (error) {
      console.error('Error submitting feedback:', error);
    }
  };

  // Futuristic Styles for React Select
  const customSelectStyles = {
    control: (base, state) => ({
      ...base,
      backgroundColor: 'rgba(18, 21, 25, 0.4)',
      borderColor: state.isFocused ? '#448c74' : 'rgba(68, 140, 116, 0.2)',
      color: 'white',
      backdropFilter: 'blur(4px)',
      boxShadow: state.isFocused ? '0 0 15px rgba(68, 140, 116, 0.2)' : 'none',
      '&:hover': {
        borderColor: '#448c74'
      }
    }),
    menu: (base) => ({
      ...base,
      backgroundColor: '#121519',
      borderColor: '#448c74',
      borderWidth: '1px',
      backdropFilter: 'blur(12px)',
      boxShadow: '0 0 20px rgba(68, 140, 116, 0.2)',
      zIndex: 50
    }),
    option: (base, state) => ({
      ...base,
      backgroundColor: state.isFocused ? '#448c74' : 'transparent',
      color: 'white',
      '&:hover': {
        backgroundColor: 'rgba(68, 140, 116, 0.8)',
        color: 'white'
      }
    }),
    multiValue: (base) => ({
      ...base,
      backgroundColor: 'rgba(68, 140, 116, 0.2)',
      border: '1px solid rgba(68, 140, 116, 0.3)',
      borderRadius: '2px'
    }),
    multiValueLabel: (base) => ({
      ...base,
      color: '#448c74',
      fontWeight: 'bold'
    }),
    singleValue: (base) => ({ ...base, color: 'white' }),
    input: (base) => ({ ...base, color: 'white' }),
    placeholder: (base) => ({ ...base, color: 'rgba(255, 255, 255, 0.5)' })
  };

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-primary selection:bg-accent selection:text-white font-sans text-white">
      {/* Top Bar */}
      <div className="h-16 border-b border-accent/20 bg-primary/90 backdrop-blur-md flex items-center px-8 justify-between z-10 shrink-0 shadow-neon">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded bg-gradient-to-br from-accent to-primary border border-accent flex items-center justify-center shadow-neon">
            <span className="text-white font-bold font-mono">P</span>
          </div>
          <h1 className="text-xl font-bold tracking-tight text-white">
            PROP<span className="text-accent">GPT</span>
          </h1>
        </div>
        <div className="px-3 py-1 rounded border border-accent/30 text-accent text-xs font-mono tracking-widest bg-accent-dim">
          SYSTEM_ONLINE
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <div className="w-80 glass-panel border-r border-accent/20 flex flex-col z-20 shrink-0">
          <div className="p-6 space-y-8 overflow-y-auto flex-1 custom-scrollbar">

            <div className="space-y-6">
              <h2 className="text-xs font-bold text-accent uppercase tracking-[0.2em] flex items-center gap-2 border-b border-accent/20 pb-2">
                PARAMETERS
              </h2>

              <div className="space-y-2">
                <label className="text-xs text-white/70 font-mono uppercase">Comparison Type</label>
                <Select
                  value={comparisonTypes.find(t => t.value === comparisonType)}
                  onChange={(option) => setComparisonType(option.value)}
                  options={comparisonTypes}
                  styles={customSelectStyles}
                  isSearchable={false}
                />
              </div>

              <div className="space-y-2">
                <label className="text-xs text-white/70 font-mono uppercase">
                  Targets <span className="text-accent ml-1">[MAX 5]</span>
                </label>
                <AsyncSelect
                  key={comparisonType}
                  cacheOptions
                  defaultOptions
                  loadOptions={loadOptions}
                  isMulti
                  value={selectedItems}
                  onChange={(items) => items && items.length <= 5 ? setSelectedItems(items) : null}
                  styles={customSelectStyles}
                  placeholder="SEARCH DATABASE..."
                  noOptionsMessage={() => "NO DATA FOUND"}
                  loadingMessage={() => "SCANNING..."}
                />
              </div>

              <div className="space-y-2">
                <label className="text-xs text-white/70 font-mono uppercase">Metrics</label>
                <Select
                  isMulti
                  value={categoryOptions.filter(c => categories.includes(c.value))}
                  onChange={(selected) => setCategories(selected.map(s => s.value))}
                  options={categoryOptions}
                  styles={customSelectStyles}
                  placeholder="SELECT METRICS..."
                />
              </div>
            </div>

            <div className="pt-6 border-t border-accent/20">
              <h3 className="text-xs font-bold text-accent uppercase tracking-[0.2em] mb-4">DIAGNOSTICS</h3>
              <div className="bg-primary/50 border border-accent/20 p-4 space-y-2 font-mono text-[10px]">
                <div className="flex justify-between">
                  <span className="text-white/60">MODEL</span>
                  <span className="text-accent">GPT-4o</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/60">VECTOR_DB</span>
                  <span className="text-accent">ONLINE</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/60">CACHE_LAYER</span>
                  <span className="text-white">ACTIVE</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Chat Area */}
        <div className="flex-1 flex flex-col relative bg-primary">
          <div className="flex-1 overflow-y-auto p-6 space-y-8 custom-scrollbar scroll-smooth relative">
            {messages.length === 0 && (
              <div className="absolute inset-0 flex flex-col items-center justify-center p-8 text-center opacity-70 pointer-events-none">
                <div className="w-24 h-24 rounded-full border border-accent/30 flex items-center justify-center mb-6 shadow-neon animate-pulse-glow bg-accent-dim">
                  <div className="w-16 h-16 rounded-full bg-accent flex items-center justify-center text-primary font-bold text-2xl">
                    AI
                  </div>
                </div>
                <h2 className="text-2xl font-bold text-white tracking-widest mb-2 font-mono">PROPGPT TERMINAL</h2>
                <p className="text-accent text-sm font-mono">Awaiting Input Sequence...</p>
              </div>
            )}

            {messages.map((message, index) => (
              <div key={index} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in`}>
                <div className={`max-w-4xl w-full ${message.role === 'user' ? 'ml-12' : 'mr-12'}`}>

                  <div className={`flex items-center gap-2 mb-2 ${message.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                    <div className={`w-6 h-6 flex items-center justify-center text-[10px] font-bold border rounded-sm font-mono ${message.role === 'user'
                        ? 'bg-transparent border-white text-white'
                        : 'bg-accent border-accent text-primary'
                      }`}>
                      {message.role === 'user' ? 'USR' : 'SYS'}
                    </div>
                    <span className="text-[10px] text-white/40 uppercase tracking-widest px-1 font-mono">
                      {message.role === 'user' ? 'QUERY' : 'RESPONSE'} // {new Date().toLocaleTimeString()}
                    </span>
                    {message.metadata?.cache_hit && (
                      <span className="text-[10px] px-1 py-0.5 bg-accent/20 text-accent border border-accent/50 font-mono">
                        CACHED
                      </span>
                    )}

                    {message.isStreaming && streamStatus && (
                      <span className="text-[10px] text-accent font-mono animate-pulse flex items-center gap-2">
                        <span className="w-1 h-1 bg-accent rounded-full"></span>
                        {streamStatus}
                      </span>
                    )}
                  </div>

                  <div className={`
                    p-6 relative border
                    ${message.role === 'user'
                      ? 'bg-white/5 border-white/10 rounded-xl rounded-tr-none'
                      : 'bg-primary border-accent/30 rounded-xl rounded-tl-none shadow-neon' // Glass/Neon effect
                    }
                    ${message.isError ? 'border-accent border-dashed opacity-80' : ''}
                  `}>
                    {message.role === 'user' ? (
                      <p className="text-lg text-white font-light">{message.content}</p>
                    ) : (
                      <div className="prose prose-invert max-w-none">
                        {message.content ? (
                          <div className="markdown-content">
                            <ReactMarkdown>{message.content}</ReactMarkdown>
                          </div>
                        ) : (
                          <div className="flex items-center gap-1 h-6 px-1">
                            <div className="w-1.5 h-1.5 bg-accent typing-dot"></div>
                            <div className="w-1.5 h-1.5 bg-accent typing-dot"></div>
                            <div className="w-1.5 h-1.5 bg-accent typing-dot"></div>
                          </div>
                        )}

                        {message.metadata && (
                          <div className="mt-8 pt-4 border-t border-accent/20 flex flex-wrap gap-4 text-[10px] text-white/50 font-mono">
                            <div className="flex items-center gap-2">
                              <span>TOKENS:</span>
                              <span className="text-accent">{message.metadata.total_tokens}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <span>MODEL:</span>
                              <span className="text-accent">{message.metadata.response_model}</span>
                            </div>

                            <div className="ml-auto flex gap-2">
                              <button
                                onClick={() => handleFeedback('thumbs_up')}
                                className="hover:text-accent transition-colors"
                              >
                                [ACKNOWLEDGE]
                              </button>
                              <button
                                onClick={() => handleFeedback('thumbs_down')}
                                className="hover:text-white transition-colors"
                              >
                                [REJECT]
                              </button>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          <div className="p-6 pt-2 shrink-0 z-20">
            <div className="glass-panel p-1 rounded-none border border-accent/40 shadow-neon-strong">
              <form onSubmit={handleSubmitQuery} className="flex gap-0 relative bg-primary">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder={selectedItems.length === 0 ? "INITIALIZE DATA SELECTION..." : "ENTER COMMAND..."}
                  className="flex-1 bg-transparent border-none text-white placeholder-white/30 px-4 py-4 focus:ring-0 text-base font-mono tracking-wide"
                  disabled={loading || selectedItems.length === 0}
                />
                <button
                  type="submit"
                  disabled={loading || !query.trim() || selectedItems.length === 0}
                  className="px-8 bg-accent text-white font-bold tracking-widest hover:bg-white hover:text-primary transition-all disabled:opacity-50 disabled:cursor-not-allowed font-mono text-xs uppercase"
                >
                  Execute
                </button>
              </form>
            </div>
            <div className="flex justify-between items-center mt-2 px-1">
              <span className="text-[10px] text-accent/60 font-mono tracking-widest">
                SECURE CONNECTION // ENCRYPTED
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
