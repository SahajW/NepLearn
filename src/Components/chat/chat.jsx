import React, { useState, useEffect } from "react";
import "./chat.css";

export const Chat = () => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");

    useEffect(() => {
        if (messages.length === 0) {
            setMessages([
                {
                    role: "bot",
                    text: "👋 Hi! I'm Nep-Learn, your AI study buddy.\nWhat topic are you studying today?",
                },
            ]);
        }
    }, []);


    /*const handleSend = () => {
        if (!input.trim()) return;

        const userMessage = {
            role: "user",
            text: input,
        };

        const botMessage = {
            role: "bot",
            text: "This is a sample chatbot reply.",
        };

        setMessages([...messages, userMessage, botMessage]);
        setInput("");
    };*/

    const handleSend = async () => {
        if (!input.trim()) return;

        const userMessage = { role: "user", text: input };
        setMessages([...messages, userMessage]); // show user's message immediately
        setInput("");

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
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === "Enter") {
            handleSend();
        }
    };

    return (
        <div className="chat-page-container">
            {/* Nav-bar */}
            <div className="navbar">
                <div className="nav-text">Nep-Learn</div>
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
                </div>

                {/* input-area */}
                <div className="chat-input-area">
                    <input
                        type="text"
                        placeholder="Type your message..."
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                    />
                    <button onClick={handleSend}>Send</button>
                </div>
            </div>
        </div>
    );
};
