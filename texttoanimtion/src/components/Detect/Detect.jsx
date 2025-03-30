import React, { useState, useRef, useEffect, useCallback } from "react";
import './a.css'
import { v4 as uuidv4 } from "uuid";
import { FilesetResolver, GestureRecognizer } from "@mediapipe/tasks-vision";
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";
import { HAND_CONNECTIONS } from "@mediapipe/hands";
import Webcam from "react-webcam";
import { SignImageData } from "../../data/SignImageData";
import { useDispatch, useSelector } from "react-redux";
import { addSignData } from "../../redux/actions/signdataaction";
import ProgressBar from "./ProgressBar/ProgressBar";


let startTime = "";

const Detect = () => {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [webcamRunning, setWebcamRunning] = useState(false);
  const [gestureOutput, setGestureOutput] = useState("");
  const [gestureRecognizer, setGestureRecognizer] = useState(null);
  const [runningMode, setRunningMode] = useState("IMAGE");
  const [progress, setProgress] = useState(0);

  // Dictionary of common English words
  const commonEnglishWords = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I", "it", "for", "not", "on", "with", 
    "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", 
    "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up", 
    "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", 
    "no", "just", "him", "know", "take", "people", "into", "year", "your", "good", "some", "could", 
    "them", "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think", 
    "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even", 
    "new", "want", "because", "any", "these", "give", "day", "most", "us", "hi", "hello", "thanks", 
    "please", "help", "need", "sign", "language", "word", "sentence"
  ];

  // Function to check if a word is in our dictionary
  const isCommonEnglishWord = (word) => {
    if (!word || word.length < 2) return false;
    return commonEnglishWords.includes(word.toLowerCase());
  };

  // States for word formation
  const [gestureBuffer, setGestureBuffer] = useState([]);
  const [formedWords, setFormedWords] = useState([]);
  const [currentSentence, setCurrentSentence] = useState("");
  const [lastDetectedTime, setLastDetectedTime] = useState(null);
  const [lastDetectedGesture, setLastDetectedGesture] = useState("");
  const bufferTimeoutRef = useRef(null);
  const characterDelayRef = useRef(null);
  const requestRef = useRef();
  const [detectedData, setDetectedData] = useState([]);
  const user = useSelector((state) => state.auth?.user);
  const { accessToken } = useSelector((state) => state.auth);
  const dispatch = useDispatch();
  const [currentImage, setCurrentImage] = useState(null);

  // Process gesture buffer and form words
  const processGestureBuffer = useCallback(() => {
    if (gestureBuffer.length === 0) return;
    const possibleWord = gestureBuffer.join("").toLowerCase();
    
    // Check if the current buffer forms a valid word
    const isValidWord = isCommonEnglishWord(possibleWord);
    
    if (isValidWord) {
      // Check for repetition - don't add if it's the same as the last word
      if (formedWords.length === 0 || formedWords[formedWords.length - 1] !== possibleWord) {
        // Add to formed words and update the sentence
        setFormedWords(prev => [...prev, possibleWord]);
        setCurrentSentence(prev => prev ? `${prev} ${possibleWord}` : possibleWord);
        
        // Save to local storage
        const savedWords = JSON.parse(localStorage.getItem('signLanguageWords') || '[]');
        localStorage.setItem('signLanguageWords', JSON.stringify([...savedWords, possibleWord]));
      }
      // Clear the buffer after forming a word
      setGestureBuffer([]);
    } else {
      // If not a valid word, check if it's a prefix of any word in the dictionary
      const matchingWords = commonEnglishWords.filter(word => word.startsWith(possibleWord));
      if (matchingWords.length > 0) {
        // Complete the word with the most likely match
        const mostLikelyWord = matchingWords[0];
        
        // Add the completed word to formed words and update sentence
        setFormedWords(prev => [...prev, mostLikelyWord]);
        setCurrentSentence(prev => prev ? `${prev} ${mostLikelyWord}` : mostLikelyWord);
        
        // Save to local storage
        const savedWords = JSON.parse(localStorage.getItem('signLanguageWords') || '[]');
        localStorage.setItem('signLanguageWords', JSON.stringify([...savedWords, mostLikelyWord]));
        
        // Clear the buffer
        setGestureBuffer([]);
      }
    }
  }, [gestureBuffer, formedWords]);

  // Handle gesture timeout (consider word complete)
  const handleGestureTimeout = useCallback(() => {
    if (gestureBuffer.length > 0) {
      processGestureBuffer();
    }
  }, [gestureBuffer, processGestureBuffer]);

  // Effect to trigger word formation when a pause in gestures is detected
  useEffect(() => {
    if (bufferTimeoutRef.current) {
      clearTimeout(bufferTimeoutRef.current);
    }
    if (gestureBuffer.length > 0) {
      bufferTimeoutRef.current = setTimeout(handleGestureTimeout, 3000); // 3-second timeout
    }
    return () => {
      if (bufferTimeoutRef.current) {
        clearTimeout(bufferTimeoutRef.current);
      }
    };
  }, [gestureBuffer, handleGestureTimeout]);

  // Clear memory on component unmount
  useEffect(() => {
    return () => {
      if (bufferTimeoutRef.current) {
        clearTimeout(bufferTimeoutRef.current);
      }
      if (characterDelayRef.current) {
        clearTimeout(characterDelayRef.current);
      }
    };
  }, []);

  useEffect(() => {
    let intervalId;
    if (webcamRunning) {
      intervalId = setInterval(() => {
        const randomIndex = Math.floor(Math.random() * SignImageData.length);
        const randomImage = SignImageData[randomIndex];
        setCurrentImage(randomImage);
      }, 5000);
    }
    return () => clearInterval(intervalId);
  }, [webcamRunning]);

  if (process.env.NODE_ENV === "development" || process.env.NODE_ENV === "production") {
    console.log = function () {};
  }

  const addCharacterToBuffer = useCallback((character) => {
    // Cancel any existing delay timer
    if (characterDelayRef.current) {
      clearTimeout(characterDelayRef.current);
    }
    
    // Set a delay before actually adding the character
    characterDelayRef.current = setTimeout(() => {
      setGestureBuffer(prev => {
        const newBuffer = [...prev, character];
        const possibleWord = newBuffer.join("").toLowerCase();
        
        // Check if the new buffer forms a valid word
        if (isCommonEnglishWord(possibleWord)) {
          // If it's a valid word, process it after a brief pause
          setTimeout(() => processGestureBuffer(), 100);
        } else {
          // Check if it's a prefix of any word in the dictionary
          const matchingWords = commonEnglishWords.filter(word => word.startsWith(possibleWord));
          if (matchingWords.length > 0 && possibleWord.length >= 3) {
            // If it's a prefix of 3+ characters, suggest the word
            console.log(`Suggested word: ${matchingWords[0]}`);
            
            // Optionally auto-complete after a certain length
            if (possibleWord.length >= 4) {
              setTimeout(() => {
                setFormedWords(prev => [...prev, matchingWords[0]]);
                setCurrentSentence(prev => prev ? `${prev} ${matchingWords[0]}` : matchingWords[0]);
                setGestureBuffer([]);
              }, 1000);
            }
          }
        }
        return newBuffer;
      });
      
      setLastDetectedGesture(character);
      
      // Reset the timeout for word formation
      if (bufferTimeoutRef.current) {
        clearTimeout(bufferTimeoutRef.current);
      }
      bufferTimeoutRef.current = setTimeout(handleGestureTimeout, 3000);
    }, 500); // 500ms delay before registering character
  }, [handleGestureTimeout, processGestureBuffer]);

  const predictWebcam = useCallback(() => {
    if (runningMode === "IMAGE") {
      setRunningMode("VIDEO");
      gestureRecognizer.setOptions({ runningMode: "VIDEO" });
    }
    
    let nowInMs = Date.now();
    const results = gestureRecognizer.recognizeForVideo(
      webcamRef.current.video,
      nowInMs
    );
    
    const canvasCtx = canvasRef.current.getContext("2d");
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    
    const videoWidth = webcamRef.current.video.videoWidth;
    const videoHeight = webcamRef.current.video.videoHeight;
    
    // Set video width
    webcamRef.current.video.width = videoWidth;
    webcamRef.current.video.height = videoHeight;
    
    // Set canvas height and width
    canvasRef.current.width = videoWidth;
    canvasRef.current.height = videoHeight;
    
    // Draw the results on the canvas, if any.
    if (results.landmarks) {
      for (const landmarks of results.landmarks) {
        drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
          color: "#00FF00",
          lineWidth: 5,
        });
        drawLandmarks(canvasCtx, landmarks, {
          color: "#FF0000",
          lineWidth: 2
        });
      }
    }
    
    if (results.gestures.length > 0) {
      const detectedGesture = results.gestures[0][0].categoryName;
      const currentTime = Date.now();
      const confidenceScore = results.gestures[0][0].score;
      
      setDetectedData((prevData) => [
        ...prevData,
        { SignDetected: detectedGesture },
      ]);
      
      setGestureOutput(detectedGesture);
      setProgress(Math.round(parseFloat(confidenceScore) * 100));
      
      // Process for word formation - only if it's a single character gesture and confidence score is high enough
      if (detectedGesture && detectedGesture.length === 1 && /[a-zA-Z]/.test(detectedGesture) && confidenceScore > 0.6) {
        // Only accept gestures with good confidence
        if ((!lastDetectedTime || detectedGesture !== lastDetectedGesture || (currentTime - lastDetectedTime) > 1500) && webcamRunning) {
          addCharacterToBuffer(detectedGesture);
          setLastDetectedTime(currentTime);
        }
      } else if (detectedGesture === "SPACE" && confidenceScore > 0.6) {
        // Handle space gesture to complete current word and add space
        processGestureBuffer();
        setCurrentSentence(prev => `${prev} `);
      }
    } else {
      setGestureOutput("");
      setProgress("");
    }
    
    if (webcamRunning === true) {
      requestRef.current = requestAnimationFrame(predictWebcam);
    }
  }, [webcamRunning, runningMode, gestureRecognizer, lastDetectedTime, lastDetectedGesture, processGestureBuffer, addCharacterToBuffer]);

  const animate = useCallback(() => {
    requestRef.current = requestAnimationFrame(animate);
    predictWebcam();
  }, [predictWebcam]);

  const enableCam = useCallback(() => {
    if (!gestureRecognizer) {
      alert("Please wait for gestureRecognizer to load");
      return;
    }
    
    if (webcamRunning === true) {
      setWebcamRunning(false);
      cancelAnimationFrame(requestRef.current);
      setCurrentImage(null);
      
      const endTime = new Date();
      const timeElapsed = ((endTime.getTime() - startTime.getTime()) / 1000).toFixed(2);
      
      // Remove empty values
      const nonEmptyData = detectedData.filter(
        (data) => data.SignDetected !== "" && data.DetectedScore !== ""
      );
      
      //to filter continous same signs in an array
      const resultArray = [];
      let current = nonEmptyData[0];
      for (let i = 1; i < nonEmptyData.length; i++) {
        if (nonEmptyData[i].SignDetected !== current.SignDetected) {
          resultArray.push(current);
          current = nonEmptyData[i];
        }
      }
      resultArray.push(current);
      
      //calculate count for each repeated sign
      const countMap = new Map();
      for (const item of resultArray) {
        const count = countMap.get(item.SignDetected) || 0;
        countMap.set(item.SignDetected, count + 1);
      }
      
      const sortedArray = Array.from(countMap.entries()).sort(
        (a, b) => b[1] - a[1]
      );
      
      const outputArray = sortedArray
        .slice(0, 5)
        .map(([sign, count]) => ({ SignDetected: sign, count }));
      
      // object to send to action creator
      const data = {
        signsPerformed: outputArray,
        id: uuidv4(),
        username: user?.name,
        userId: user?.userId,
        createdAt: String(endTime),
        secondsSpent: Number(timeElapsed),
        formedWords: formedWords,
        sentence: currentSentence
      };
      
      dispatch(addSignData(data));
      setDetectedData([]);
      setGestureBuffer([]);
      setFormedWords([]);
      setCurrentSentence("");
    } else {
      setWebcamRunning(true);
      startTime = new Date();
      requestRef.current = requestAnimationFrame(animate);
    }
  }, [webcamRunning, gestureRecognizer, animate, detectedData, user?.name, user?.userId, dispatch, formedWords, currentSentence]);

  // Reset the sentence
  const resetSentence = () => {
    setGestureBuffer([]);
    setFormedWords([]);
    setCurrentSentence("");
  };

  useEffect(() => {
    async function loadGestureRecognizer() {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      );
      const recognizer = await GestureRecognizer.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: 'https://firebasestorage.googleapis.com/v0/b/reboostify-f18db.appspot.com/o/sign_language_recognizer_25-04-2023.task?alt=media&token=6848c37c-7b5d-4bf3-9aba-0c9b254ba305',
        },
        numHands: 2,
        runningMode: runningMode,
      });
      setGestureRecognizer(recognizer);
    }
    loadGestureRecognizer();
  }, [runningMode]);

  return (
<>
  
    <div className="gesture-output">
      <span>{gestureOutput}</span>
     
    </div>

  <div className="webcam-container">
    <Webcam
      ref={webcamRef}
      style={{
        position: "absolute",
        marginLeft: "auto",
        marginRight: "auto",
        left: 0,
        right: 0,
        textAlign: "center",
        zIndex: -1,
        width: 640,
        height: 480,
      }}
    />
    <canvas
      ref={canvasRef}
      style={{
        position: "absolute",
        marginLeft: "auto",
        marginRight: "auto",
        left: 0,
        right: 0,
        textAlign: "center",
        zIndex: -1,
        width: 640,
        height: 480,
      }}
    />
  </div>
  <div className="controls">
    <button onClick={enableCam}>{webcamRunning ? "STOP" : "START"}</button>
  </div>
  <div className="word-formation">
    <h2>Formed Words:</h2>
    <ul>
      {formedWords.map((word, index) => (
        <li key={index}>{word}</li>
      ))}
    </ul>
    <h2>Current Sentence:</h2>
    <p>{currentSentence}</p>
    <button onClick={resetSentence}>Reset Sentence</button>
  </div>

</>

  );
};

export default Detect;
