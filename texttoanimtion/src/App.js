import React, { useState, useEffect, useRef } from 'react';
import Good from "./gifs/good.gif" 
import Morning from "./gifs/morning.gif"
import Hello from "./gifs/hello.gif"
import You from "./gifs/you.gif" 

const SignLanguagePlayer = () => {
  const [inputText, setInputText] = useState('');
  const [currentWord, setCurrentWord] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [filteredWords, setFilteredWords] = useState([]);
  const [showFiltered, setShowFiltered] = useState(false);
  const [loopCount, setLoopCount] = useState(0);
  const gifQueueRef = useRef([]);
  const originalQueueRef = useRef([]);
  const timeoutRef = useRef(null);

  // Define our sign language GIFs
  const signLanguageGifs = [
    { word: "good", gifPath: Good },
    { word: "morning", gifPath: Morning },
    { word: "hello", gifPath: Hello },
    { word: "you", gifPath: You},
  ];
  
  // Words to filter out
  const filterWords = ["it", "the", "a", "an", "in", "and", "or", "but", "is", "are"];

  const processText = () => {
    if (isPlaying) return;
    
    const text = inputText.trim().toLowerCase();
    if (!text) return;
    
    // Split text into words
    let words = text.split(/\s+/);
    
    // Filter out words
    const originalWords = [...words];
    words = words.filter(word => !filterWords.includes(word));
    
    // Check if words were filtered
    const filtered = originalWords.filter(word => filterWords.includes(word));
    setFilteredWords(filtered);
    setShowFiltered(filtered.length > 0);
    
    // Create queue of GIFs to play
    const queue = [];
    words.forEach(word => {
      const gifData = signLanguageGifs.find(item => item.word === word);
      if (gifData) {
        queue.push(gifData);
      }
    });
    
    // Store the original queue for looping
    originalQueueRef.current = [...queue];
    gifQueueRef.current = [...queue];
    
    // Reset loop count
    setLoopCount(0);
    
    // Start playing if we have GIFs in the queue
    if (queue.length > 0) {
      setIsPlaying(true);
      playNextGif();
    } else {
      setCurrentWord('');
    }
  };

  const playNextGif = () => {
    if (gifQueueRef.current.length === 0) {
      // Queue is empty, reset it and start over
      gifQueueRef.current = [...originalQueueRef.current];
      setLoopCount(prevCount => prevCount + 1);
    }
    
    // Get the next GIF from the queue
    const nextGif = gifQueueRef.current.shift();
    
    // Display the GIF
    setCurrentWord(nextGif.word);
    
    // After 2 seconds, play the next GIF
    timeoutRef.current = setTimeout(playNextGif, 2000);
  };

  const stopPlayback = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    setIsPlaying(false);
    setCurrentWord('');
    gifQueueRef.current = [];
  };

  // Clean up timeout on unmount
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      processText();
    }
  };

  return (
    <div style={{ backgroundColor: '#f7fafc', minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '16px' }}>
      <div style={{ maxWidth: '640px', width: '100%', backgroundColor: '#ffffff', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)', padding: '24px', margin: 'auto' }}>
        <h1 style={{ fontSize: '24px', fontWeight: 'bold', textAlign: 'center', marginBottom: '24px', color: '#2d3748' }}></h1>
        
        <div style={{ marginBottom: '24px' }}>
          <label htmlFor="textInput" style={{ fontSize: '14px', fontWeight: 'medium', color: '#4a5568', marginBottom: '8px', display: 'block' }}>
          
          </label>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <input 
              type="text" 
              id="textInput" 
              style={{ flexGrow: '1', padding: '8px 16px', border: '1px solid #e2e8f0', borderRadius: '4px', outline: 'none', fontSize: '14px' }}
              placeholder=""
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyPress={handleKeyPress}
            />
            <div style={{ display: 'flex', gap: '8px' }}>
              <button 
                style={{ 
                  backgroundColor: isPlaying ? '#cbd5e0' : '#3182ce', 
                  color: 'white', 
                  padding: '8px 16px', 
                  borderRadius: '4px', 
                  cursor: isPlaying ? 'not-allowed' : 'pointer', 
                  transition: 'background-color 0.2s',
                  flexGrow: 1
                }}
                onClick={processText}
                disabled={isPlaying}
              >
                Play
              </button>
              {isPlaying && (
                <button 
                  style={{ 
                    backgroundColor: '#e53e3e', 
                    color: 'white', 
                    padding: '8px 16px', 
                    borderRadius: '4px', 
                    cursor: 'pointer', 
                    transition: 'background-color 0.2s',
                    flexGrow: 1
                  }}
                  onClick={stopPlayback}
                >
                  Stop
                </button>
              )}
            </div>
          </div>
          {showFiltered && (
            <p style={{ marginTop: '8px', fontSize: '12px', color: '#6b7280' }}>
              Filtered words: {filteredWords.join(", ")}
            </p>
          )}
        </div>
        
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ width: '256px', height: '256px', backgroundColor: '#edf2f7', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            {isPlaying && currentWord ? (
              <>
                <img 
                  src={signLanguageGifs.find(item => item.word === currentWord)?.gifPath} 
                  alt={`${currentWord} sign`} 
                  style={{ maxWidth: '100%', maxHeight: '100%' }} 
                />
                <div style={{ position: 'absolute', bottom: '8px', right: '8px', backgroundColor: 'rgba(0,0,0,0.5)', color: 'white', padding: '2px 6px', borderRadius: '4px', fontSize: '12px' }}>
                  Loop: {loopCount}
                </div>
              </>
            ) : (
              <p style={{ color: '#6b7280' }}>
                {originalQueueRef.current.length > 0 && !isPlaying 
                  ? "Paused" 
                  : "Sign language GIFs will appear here"}
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SignLanguagePlayer;