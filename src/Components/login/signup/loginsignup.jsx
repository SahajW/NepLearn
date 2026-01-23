import React, { useState } from "react";
import "../loginsignup.css";
import { useNavigate } from 'react-router-dom';

import user_icon from "../../Assets/user.png";
import email_icon from "../../Assets/email.png";
import lock_icon from "../../Assets/padlock.png";

export const Loginsignup = () => {
  const navigate = useNavigate();
  const [isLogin, setIsLogin] = useState(true); // Start with Login page
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
    confirmPassword: ""
  });
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [loading, setLoading] = useState(false);

  const API_URL = "http://localhost:8000";

  // NEW: Helper function to extract error message
  const getErrorMessage = (data) => {
    // If detail is a string, return it
    if (typeof data.detail === 'string') {
      return data.detail;
    }
    
    // If detail is an array (Pydantic validation errors)
    if (Array.isArray(data.detail)) {
      // Extract all error messages and join them
      return data.detail
        .map(err => err.msg || JSON.stringify(err))
        .join('. ');
    }
    
    // If detail is an object
    if (typeof data.detail === 'object') {
      return JSON.stringify(data.detail);
    }
    
    // Fallback
    return "An error occurred";
  };

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
    setError("");
    setSuccess("");
  };

  const handleSignup = async () => {
    // Validation
    if (!formData.username || !formData.email || !formData.password || !formData.confirmPassword) {
      setError("All fields are required");
      return;
    }

    if (formData.password !== formData.confirmPassword) {
      setError("Passwords do not match");
      return;
    }

    if (formData.password.length < 8) {
      setError("Password must be at least 8 characters");
      return;
    }
    if (!/[A-Z]/.test(formData.password)) {
      setError("Password must contain at least one uppercase letter");
      return;
    }
    if (!/[0-9]/.test(formData.password)) {
      setError("Password must contain at least one number");
      return;
    }
    if (!/[a-z]/.test(formData.password)) {
      setError("Password must contain at least one lowercase letter");
      return;
    }
    if (!/[!@#$%^&*]/.test(formData.password)) {
      setError("Password must contain at least one special character (!@#$%^&*)");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/signup`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          username: formData.username,
          email: formData.email,
          password: formData.password
        }),
      });

      const data = await response.json();

      if (response.ok) {
        setSuccess("Account created successfully! Please login.");
        setFormData({
          username: "",
          email: "",
          password: "",
          confirmPassword: ""
        });
        // Switch to login page after 2 seconds
        setTimeout(() => {
          setIsLogin(true);
          setSuccess("");
        }, 2000);
      } else {
        // ✅ FIXED: Use helper function to extract error message
        setError(getErrorMessage(data));
      }
    } catch (err) {
      setError("Network error. Please try again.");
      console.error("Signup error:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleLogin = async () => {
    // Validation
    if (!formData.email || !formData.password) {
      setError("Email and password are required");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/login`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          email: formData.email,
          password: formData.password
        }),
      });

      const data = await response.json();

      if (response.ok) {
        localStorage.setItem("token", data.access_token);
        localStorage.setItem("username", data.username);
        setSuccess("Login successful! Redirecting...");
        
        // Redirect after short delay
        setTimeout(() => {
          navigate('/home');
        }, 1000);
      } else {
        // FIXED: Use helper function to extract error message
        setError(getErrorMessage(data));
      }
    } catch (err) {
      setError("Network error. Please try again.");
      console.error("Login error:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = () => {
    if (isLogin) {
      handleLogin();
    } else {
      handleSignup();
    }
  };

  const switchMode = () => {
    setIsLogin(!isLogin);
    setError("");
    setSuccess("");
    setFormData({
      username: "",
      email: "",
      password: "",
      confirmPassword: ""
    });
  };

  return (
    <div className="container">
      <div className="header">
        <div className="text">{isLogin ? "Login" : "Sign Up"}</div>
        <div className="underline"></div>
      </div>
      
      {/* These are already correct - displaying strings */}
      {error && <div className="error-message">{error}</div>}
      {success && <div className="success-message">{success}</div>}
      
      <div className="inputs">
        {!isLogin && (
          <div className="input">
            <img src={user_icon} alt="" />
            <input 
              type="text" 
              placeholder="Username"
              name="username"
              value={formData.username}
              onChange={handleInputChange}
            />
          </div>
        )}
        
        <div className="input">
          <img src={email_icon} alt="" />
          <input 
            type="email" 
            placeholder="Email"
            name="email"
            value={formData.email}
            onChange={handleInputChange}
          />
        </div>
        
        <div className="input">
          <img src={lock_icon} alt="" />
          <input 
            type="password" 
            placeholder="Password"
            name="password"
            value={formData.password}
            onChange={handleInputChange}
          />
        </div>
        
        {!isLogin && (
          <div className="input">
            <img src={lock_icon} alt="" />
            <input 
              type="password" 
              placeholder="Confirm Password"
              name="confirmPassword"
              value={formData.confirmPassword}
              onChange={handleInputChange}
            />
          </div>
        )}
        
        {isLogin && (
          <div className="forgot-password">
            Forgot Password? <span>Click Here!</span>
          </div>
        )}
        
        <div className="submit-container">
          <div
            className={`submit primary ${loading ? "disabled" : ""}`}
            onClick={loading ? null : handleSubmit}
          >
            {loading ? "Processing..." : (isLogin ? "Login" : "Sign Up")}
          </div>
        </div>

        <div className="switch-mode">
          {isLogin ? (
            <p>
              Don't have an account? 
              <span onClick={switchMode}> Sign Up</span>
            </p>
          ) : (
            <p>
              Already have an account? 
              <span onClick={switchMode}> Login</span>
            </p>
          )}
        </div>
      </div>
    </div>
  );
};