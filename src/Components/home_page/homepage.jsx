
// Homepage.jsx
import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "./homepage.css";

export const Homepage = () => {
  const navigate = useNavigate();
  const [username, setUsername] = useState("");

  useEffect(() => {
    // Get username from localStorage
    const storedUsername = localStorage.getItem("username");
    if (storedUsername) {
      setUsername(storedUsername);
    }
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('userData');
    localStorage.removeItem('username'); // Clear username on logout
    navigate('/');
  };

  return (
    <div className="home-container">
      {/* Navigation Bar */}
      <div className="navbar">
        
        {/* Display username */}
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '15px' }}>
          {username && (
            <span style={{ fontSize: '16px', fontWeight: '500' }}>
              👤 {username}
            </span>
          )}
          <button 
            onClick={handleLogout}
            style={{
              padding: '8px 20px',
              background: '#d9534f',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: '500'
            }}
          >
            Logout
          </button>
        </div>
      </div>

      <div className="main-content">
        <div className="box-wrapper">
          <div className="box" onClick={() => navigate('/chat')}>
            <h2>Generate Answer</h2>
          </div>
          <div className="box" onClick={() => navigate('/generate')}>
            <h2>Generate Question</h2>
          </div>
        </div>
      </div>

      <div className="footer">
        <div className="sliding-text">
          Nep Learn!    Nep Learn!    Nep Learn!    Nep Learn!    Nep Learn!    Nep Learn!    Nep Learn!    Nep Learn!    Nep Learn!    Nep Learn!    Nep Learn!    Nep Learn!    Nep Learn!
        </div>
      </div>
    </div>
  );
};