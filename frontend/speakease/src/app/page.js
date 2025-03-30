"use client"
import React, { useState, useEffect, useRef } from 'react';
import Good from "/public/gifs/good.gif" 
import Morning from "/public/gifs/morning.gif"
import Hello from "/public/gifs/hello.gif"
import You from "/public/gifs/you.gif" 
const SignLanguagePlayer = () => {
  const [inputText, setInputText] = useState('');
  const [currentWord, setCurrentWord] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [filteredWords, setFilteredWords] = useState([]);
  const [showFiltered, setShowFiltered] = useState(false);
  const gifQueueRef = useRef([]);
  const timeoutRef = useRef(null);

  // Define our sign language GIFs
  const signLanguageGifs = [
    { word: "good", gifPath:Good },
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
    
    gifQueueRef.current = queue;
    
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
      // Queue is empty, reset
      setIsPlaying(false);
      setCurrentWord('');
      return;
    }
    
    // Get the next GIF from the queue
    const nextGif = gifQueueRef.current.shift();
    
    // Display the GIF
    setCurrentWord(nextGif.word);
    
    // After 2 seconds, play the next GIF
    timeoutRef.current = setTimeout(playNextGif, 2000);
  };

  // Clean up timeout on unmount or when stopping playback
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
    <div className="bg-gray-100 min-h-screen flex flex-col items-center p-4">
      <div className="max-w-2xl w-full bg-white rounded-lg shadow-md p-6 mx-auto">
        <h1 className="text-2xl font-bold text-center mb-6 text-gray-800">Sign Language GIF Player</h1>
        
        <div className="mb-6">
          <label htmlFor="textInput" className="block text-sm font-medium text-gray-700 mb-2">
            Enter text to convert to sign language:
          </label>
          <div className="flex flex-col sm:flex-row gap-2">
            <input 
              type="text" 
              id="textInput" 
              className="flex-grow px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500" 
              placeholder="Type here (e.g. good morning hello you)"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyPress={handleKeyPress}
            />
            <button 
              className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition"
              onClick={processText}
              disabled={isPlaying}
            >
              Play
            </button>
          </div>
          {showFiltered && (
            <p className="mt-2 text-sm text-gray-500">
              Filtered words: {filteredWords.join(", ")}
            </p>
          )}
        </div>
        
        <div className="flex flex-col items-center justify-center">
          <div className="w-64 h-64 bg-gray-200 rounded-lg flex items-center justify-center">
            {isPlaying && currentWord ? (
              <img 
                src={signLanguageGifs.find(item => item.word === currentWord)?.gifPath} 
                alt={`${currentWord} sign`} 
                className="max-w-full max-h-full" 
              />
            ) : (
              <p className="text-gray-500">
                {gifQueueRef.current.length === 0 && isPlaying 
                  ? "Done" 
                  : "Sign language GIFs will appear here"}
              </p>
            )}
          </div>
          <div className="mt-4">
            {currentWord && (
              <p className="text-lg font-medium text-center">Showing: "{currentWord}"</p>
            )}
          </div>
        </div>

        <div className="mt-6">
          <h2 className="text-lg font-semibold mb-2">Available Words:</h2>
          <div className="flex flex-wrap gap-2">
            {signLanguageGifs.map((item, index) => (
              <span 
                key={index} 
                className="bg-blue-100 text-blue-800 text-xs font-medium px-2.5 py-0.5 rounded"
              >
                {item.word}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SignLanguagePlayer;