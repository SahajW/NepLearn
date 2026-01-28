/*import React, { useState } from "react";
import "./generatequestion.css";

export const Generatequestion = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const handleGenerate = async () => {
    if (!input.trim()) return;

    // user message
    const userMessage = {
      role: "user",
      text: input,
    };

    // bot response 
    /*const botMessage = {
      role: "bot",
      text: "This is a sample generated question response.",
    };

    setMessages([...messages, userMessage]);
    setInput("");
    setMessages((prev) => [...prev, userMessage]); 
    const currentInput = input;  
    setInput("");
    setLoading(true);

    try {
      const requestbody = {
        structure: {
          "SECTION B": {
            "count": 7,
            "instruction": "Attempt Any SIX Questions",
            "min_length": 30,
            "max_length": 400
          },
          "SECTION C": {
            "count": 3,
            "instruction": "Attempt Any TWO Questions",
            "min_length": 150,

          },
        },
        similarity_threshold: 0.85,
        source_filter: "mixed",
      }
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestbody),
      });

      const data = await response.json();
      const botMessage = { role: "bot", text: JSON.stringify(data.paper, null, 2) };
      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      console.error("Error sending message:", err);
      const botMessage = { role: "bot", text: "Oops! Something went wrong." };
      setMessages((prev) => [...prev, botMessage]);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      handleGenerate();
    }
  };

  return (
    <div className="home-container">
      {/* Navigation Bar *}
      <div className="navbar">
        <div className="nav-text">Nep-Learn</div>
      </div>

      {/* Chat Container *}
      <div className="chat-container">
        <div className="chat-messages">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`chat-message ${msg.role}`}
            >
              {msg.text}
            </div>
          ))}
        </div>

        {/* Input Area }
        <div className="chat-input-area">
          <input
            type="text"
            placeholder="Ask Nep-Learn to generate a question..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <button onClick={handleGenerate}>Generate</button>
        </div>
      </div>
    </div>
  );
};

*/

import React, { useState, useRef, useEffect } from "react";
import "./generatequestion.css";

export const Generatequestion = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);  // Add loading state
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleGenerate = async () => {
    if (!input.trim() || loading) return;  // Prevent multiple submissions

    const userMessage = {
      role: "user",
      text: input,
    };

    // Add user message and clear input
    setMessages((prev) => [...prev, userMessage]);  // Use functional update
    const currentInput = input;  // Save input before clearing
    setInput("");
    setLoading(true);  // Set loading state

    try {
      const requestbody = {
        structure: {
          "SECTION B": {
            "count": 7,
            "instruction": "Attempt Any SIX Questions",
            "min_length": 30,
            "max_length": 400
          },
          "SECTION C": {
            "count": 3,
            "instruction": "Attempt Any TWO Questions",
            "min_length": 150,
          },
        },
        similarity_threshold: 0.85,
        source_filter: "mixed",
      };
      
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestbody),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      const botMessage = { 
        role: "bot", 
        text: JSON.stringify(data.paper, null, 2) 
      };
      
      setMessages((prev) => [...prev, botMessage]);  // Use functional update
    } catch (err) {
      console.error("Error sending message:", err);
      const botMessage = { 
        role: "bot", 
        text: `Error: ${err.message}. Please try again.` 
      };
      setMessages((prev) => [...prev, botMessage]);  // Use functional update
    } finally {
      setLoading(false);  // Reset loading state
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !loading) {  // Prevent submit while loading
      handleGenerate();
    }
  };

  return (
    <div className="home-container">
      <div className="navbar">
        <div className="nav-text">Nep-Learn</div>
      </div>

      <div className="chat-container">
        <div className="chat-messages">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`chat-message ${msg.role}`}
            >
              <pre style={{ whiteSpace: "pre-wrap", fontFamily: "inherit" }}>
                {msg.text}
              </pre>
            </div>
          ))}
          {loading && (
            <div className="chat-message bot">
              <span className="loading-dots">Generating</span>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="chat-input-area">
          <input
            type="text"
            placeholder="Ask Nep-Learn to generate a question..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}  // Disable input while loading
          />
          <button onClick={handleGenerate} disabled={loading}>
            {loading ? "Generating..." : "Generate"}
          </button>
        </div>
      </div>
    </div>
  );
};