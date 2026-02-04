// Chat.jsx
import React, { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "./chat.css";

export const Chat = () => {
    const [messages, setMessages] = useState([]);
    const [loading, setLoading] = useState(false);
    const [input, setInput] = useState("");
    const messagesEndRef = useRef(null);
    const navigate = useNavigate();

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
        if (messages.length === 0) {
            setMessages([
                {
                    role: "bot",
                    text: "👋 Hi! I'm Nep-Learn, your AI study buddy.\nWhat topic are you studying today?",
                },
            ]);
        }
    }, [messages]);

    const handleSend = async () => {
        if (!input.trim() || loading) return;

        const userMessage = { role: "user", text: input };
        setMessages([...messages, userMessage]);
        setInput("");
        setLoading(true);

        try {
            const response = await fetch("http://localhost:8000/ask", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question: input }),
            });

            const data = await response.json();
            const botMessage = { role: "bot", text: data.answer };
            setMessages((prev) => [...prev, botMessage]);
        } catch (err) {
            console.error("Error sending message:", err);
            const botMessage = { role: "bot", text: "Oops! Something went wrong." };
            setMessages((prev) => [...prev, botMessage]);
        } finally {
            setLoading(false);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === "Enter" && !loading) {
            handleSend();
        }
    };

    return (
        <div className="chat-page-container">
            {/* Nav-bar with navigation links */}
            <div className="navbar">
                <div className="nav-text" onClick={() => navigate('/')} style={{ cursor: 'pointer' }}>
                    Nep-Learn
                </div>
                <div className="nav-links" style={{ display: 'flex', gap: '20px', marginLeft: 'auto' }}>
                    <button 
                        onClick={() => navigate('/home')} 
                        style={{ padding: '8px 16px', cursor: 'pointer', background: 'transparent', color: 'inherit', border: '1px solid currentColor', borderRadius: '4px' }}
                    >
                        Home
                    </button>
                    <button 
                        onClick={() => navigate('/generate')} 
                        style={{ padding: '8px 16px', cursor: 'pointer', background: 'transparent', color: 'inherit', border: '1px solid currentColor', borderRadius: '4px' }}
                    >
                        Generate Question
                    </button>
                </div>
            </div>

            {/* chat-body */}
            <div className="chat-wrapper">
                <div className="chat-messages">
                    {messages.length === 0 && (
                        <div className="chat-placeholder">
                            Start a conversation with Nep-Learn
                        </div>
                    )}

                    {messages.map((msg, index) => (
                        <div
                            key={index}
                            className={`chat-message ${msg.role}`}
                        >
                            {msg.text}
                        </div>
                    ))}
                    {loading && (
                        <div className="chat-message bot">
                            <span className="loading-dots">Generating answer</span>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* input-area */}
                <div className="chat-input-area">
                    <input
                        type="text"
                        placeholder="Type your message..."
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        disabled={loading}
                    />
                    <button onClick={handleSend} disabled={loading}>
                        {loading ? "Generating..." : "Generate"}
                    </button>
                </div>
            </div>
        </div>
    );
};