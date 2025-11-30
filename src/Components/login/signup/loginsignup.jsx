import React from "react";
import "../loginsignup.css";

import user_icon from "../../Assets/user.png";
import email_icon from "../../Assets/email.png";
import lock_icon from "../../Assets/padlock.png";

export const Loginsignup = () => {
  return (
    <div className="container">
      <div className="header">
        <div className="text">Sign Up</div>
        <div className="underline"></div>
      </div>
      <div className="inputs">
        <div className="input">
          <img src={user_icon} alt="" />
          <input type="text" />
        </div>
        <div className="input">
          <img src={email_icon} alt="" />
          <input type="email" />
        </div>
        <div className="input">
          <img src={lock_icon} alt="" />
          <input type="password" />
        </div>
      </div>
    </div>
  );
};
