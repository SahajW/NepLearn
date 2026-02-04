import React from "react";
import { useNavigate } from "react-router-dom";
import "./homepage.css";

export const Homepage = () => {
  const navigate = useNavigate();

  const handleLogout = () => {
    // Clear any stored authentication tokens or user data
    localStorage.removeItem('authToken');
    localStorage.removeItem('userData');
    // Add any other cleanup you need
    
    // Navigate to login page (adjust the route as needed)
    navigate('/');
  };

  return (
    <div className="home-container">

      {/* Navigation Bar */}
      <div className="navbar">
        <div className="nav-text" onClick={() => navigate('/')} style={{ cursor: 'pointer' }}>
          Nep-Learn
        </div>
        <button 
          onClick={handleLogout}
          className="logout-button"
          style={{
            marginLeft: 'auto',
            padding: '8px 20px',
            background: '#d9534f',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '14px',
            fontWeight: '500',
            transition: 'background 0.3s ease'
          }}
          onMouseEnter={(e) => e.target.style.background = '#c9302c'}
          onMouseLeave={(e) => e.target.style.background = '#d9534f'}
        >
          Logout
        </button>
      </div>

      <div className="main-content">
        <div className="box-wrapper">

          {/* Box 1: Goes to Chat */}
          <div className="box" onClick={() => navigate('/chat')}>
            <h2>Generate Answer</h2>
          </div>

          {/* Box 2: Goes to another page */}
          <div className="box" onClick={() => navigate('/generate')}>
            <h2>Generate Question</h2>
          </div>

        </div>
      </div>

      {/* Brown Sliding Footer */}
      <div className="footer">
        <div className="sliding-text">
          Nep Learn!    Nep Learn!    Nep Learn!    Nep Learn!    Nep Learn!    Nep Learn!    Nep Learn!    Nep Learn!    Nep Learn!    Nep Learn!    Nep Learn!    Nep Learn!    Nep Learn!
        </div>
      </div>

    </div>
  );
};