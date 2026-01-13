import React, { useState } from "react";
import "./generatequestion.css";

export const Generatequestion = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const handleGenerate = () => {
    if (!input.trim()) return;

    // user message
    const userMessage = {
      role: "user",
      text: input,
    };

    // bot response 
    const botMessage = {
      role: "bot",
      text: "This is a sample generated question response.",
    };

    setMessages([...messages, userMessage, botMessage]);
    setInput("");
  };

  return (
    <div className="home-container">
      {/* Navigation Bar */}
      <div className="navbar">
        <div className="nav-text">Nep-Learn</div>
      </div>

      {/* Chat Container */}
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

        {/* Input Area */}
        <div className="chat-input-area">
          <input
            type="text"
            placeholder="Ask Nep-Learn to generate a question..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
          <button onClick={handleGenerate}>Generate</button>
        </div>
      </div>
    </div>
  );
};
