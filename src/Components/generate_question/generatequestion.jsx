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


/*
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
};*/
// Generatequestion.jsx
// Generatequestion.jsx
import React, { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "./generatequestion.css";

export const Generatequestion = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const navigate = useNavigate();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    if (messages.length === 0) {
      setMessages([
        {
          role: "bot",
          text: (
            <>
              <p>👋 <strong>Hi! I'm Nep-Learn, your AI study buddy.</strong></p>
              <p>For generating questions.</p>
              <p>Use a prompt like:</p>
              <p>
                <strong>
                  Generate question paper of ("subject") for (year, eg. "2026")  and number of set ("10" ; max=10)
                  Example: Generate question paper of C-Programming for 2026 and number of set 1.
                </strong>
              </p>
            </>
          ),
        },
      ]);
    }
    scrollToBottom();
  }, [messages]);

  const formatPaper = (data) => {
    if (!data) return "❌ No data available";

    let formatted = "";

    formatted += `${"═".repeat(60)}\n`;
    formatted += `📋 QUESTION PAPER - SET ${data.set_number || 1}\n`;
    formatted += `${"═".repeat(60)}\n\n`;

    if (data.metadata) {
      if (data.metadata.subject) {
        formatted += `📚 Subject: ${data.metadata.subject}\n`;
      }
      if (data.metadata.year) {
        formatted += `📅 Year: ${data.metadata.year}\n`;
      }
      if (data.metadata.duration) {
        formatted += `⏱️  Duration: ${data.metadata.duration}\n`;
      }
      if (data.metadata.total_marks) {
        formatted += `💯 Total Marks: ${data.metadata.total_marks}\n`;
      }
      if (data.metadata.score !== undefined && data.metadata.score !== null) {
        const scoreNum = Number(data.metadata.score);
        if (!isNaN(scoreNum)) {
          formatted += `🎯 Prediction Score: ${scoreNum.toFixed(2)}\n`;
        }
      }
      formatted += `\n`;
    }

    if (data.sections) {
      Object.entries(data.sections).forEach(([sectionName, sectionData]) => {
        formatted += `${"━".repeat(60)}\n`;
        formatted += `${sectionName}\n`;
        formatted += `${"━".repeat(60)}\n\n`;

        if (sectionData.instruction) {
          formatted += `📌 ${sectionData.instruction}\n\n`;
        }

        if (sectionData.questions && Array.isArray(sectionData.questions)) {
          sectionData.questions.forEach((q, idx) => {
            formatted += `${idx + 1}. ${q.question || q.text || "Question not available"}\n`;

            const metadata = [];

            if (q.source) {
              metadata.push(`Source: ${q.source}`);
            }

            if (q.prediction_score !== undefined && q.prediction_score !== null) {
              const scoreNum = Number(q.prediction_score);
              if (!isNaN(scoreNum)) {
                metadata.push(`Score: ${scoreNum.toFixed(2)}`);
              }
            }

            if (q.marks) {
              metadata.push(`💯 ${q.marks} marks`);
            }

            if (metadata.length > 0) {
              formatted += `   ${metadata.join('  •  ')}\n`;
            }

            formatted += `\n`;
          });
        }
      });
    }

    formatted += `${"━".repeat(60)}\n`;
    formatted += `🎓 Generated by Nep-Learn\n`;
    formatted += `${"━".repeat(60)}\n\n`;

    return formatted;
  };

  const handleGenerate = async () => {
    if (!input.trim() || loading) return;

    const userMessage = {
      role: "user",
      text: input,
    };

    setMessages((prev) => [...prev, userMessage]);
    const userInput = input;
    setInput("");
    setLoading(true);

    try {
      const requestbody = {
        user_input: userInput,
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
      };

      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestbody),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      // Check if we have multiple sets
      if (data.papers && Array.isArray(data.papers)) {
        // Format all sets together
        let allFormattedText = `${"═".repeat(60)}\n`;
        allFormattedText += `📚 TOTAL SETS GENERATED: ${data.total_sets}\n`;
        allFormattedText += `${"═".repeat(60)}\n\n`;

        data.papers.forEach((paper, index) => {
          allFormattedText += formatPaper(paper);
          if (index < data.papers.length - 1) {
            allFormattedText += `\n${"═".repeat(60)}\n\n`;
          }
        });

        const botMessage = {
          role: "bot",
          text: allFormattedText
        };

        setMessages((prev) => [...prev, botMessage]);
      } else {
        // Single set (backward compatibility)
        const formattedText = formatPaper(data.paper || data);
        const botMessage = {
          role: "bot",
          text: formattedText
        };
        setMessages((prev) => [...prev, botMessage]);
      }
    } catch (err) {
      console.error("Error sending message:", err);
      const botMessage = {
        role: "bot",
        text: `❌ Error: ${err.message}`
      };
      setMessages((prev) => [...prev, botMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !loading) {
      handleGenerate();
    }
  };


  return (
    <div className="home-container">
      {/* Navigation Bar with links */}
      <div className="navbar">
        <div className="nav-text" onClick={() => navigate('/home')} style={{ cursor: 'pointer' }}>
          Nep-Learn
        </div>
        <div className="nav-links" style={{ display: 'flex', gap: '20px', marginLeft: 'auto', alignItems: 'center' }}>
          <button 
            onClick={() => navigate('/home')} 
            style={{ padding: '8px 16px', cursor: 'pointer', background: 'transparent', color: 'inherit', border: '1px solid currentColor', borderRadius: '4px' }}
          >
            Home
          </button>
          <button 
            onClick={() => navigate('/chat')} 
            style={{ padding: '8px 16px', cursor: 'pointer', background: 'transparent', color: 'inherit', border: '1px solid currentColor', borderRadius: '4px' }}
          >
            Generate Answer
          </button>
        </div>
      </div>

      <div className="chat-container">
        <div className="chat-messages">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`chat-message ${msg.role}`}
            >
              <pre style={{
                whiteSpace: "pre-wrap",
                fontFamily: "inherit",
                margin: 0,
                lineHeight: "1.7"
              }}>
                {msg.text}
              </pre>
            </div>
          ))}
          {loading && (
            <div className="chat-message bot">
              <span className="loading-dots">Generating your question paper(s)</span>
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
            disabled={loading}
          />
          <button onClick={handleGenerate} disabled={loading}>
            {loading ? "Generating..." : "Generate"}
          </button>
        </div>
      </div>
    </div>
  );
};