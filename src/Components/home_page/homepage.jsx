import React from "react";
import { useNavigate } from "react-router-dom"; // Import hook for navigation
import "./homepage.css";

export const Homepage = () => {
  const navigate = useNavigate();

  return (
    <div className="home-container">

      {/* Navigation Bar */}
      <div className="navbar">
        <div className="nav-text">Nep-Learn</div>
      </div>

      <div className="main-content">
        <div className="box-wrapper">

          {/* Box 1: Goes to Chat */}
          <div className="box" onClick={() => navigate('/chat')}>
            <h2>Chat</h2>
          </div>

          {/* Box 2: Goes to another page */}
          <div className="box" onClick={() => navigate('/other-page')}>
            <h2>Generate Question</h2>
          </div>

        </div>
      </div>

      {/* 3. Brown Sliding Footer */}
      <div className="footer">
        <div className="sliding-text">
          Nep Learn!    Nep Learn!    Nep Learn!    Nep Learn!    Nep Learn!    Nep Learn!    Nep Learn!    Nep Learn!    Nep Learn!    Nep Learn!    Nep Learn!    Nep Learn!    Nep Learn!
        </div>
      </div>

    </div>
  );
};