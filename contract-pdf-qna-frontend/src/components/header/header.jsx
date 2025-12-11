import React from "react";
import bellIcon from "../../assets/bell.svg";
import userIcon from "../../assets/user.svg";
import "./header.scss";

const Header = ({ userIconImage }) => {
  return (
    <div className="header_section">
      <div className="title">AHS Customer Representative Copilot</div>
      <div className="icons">
        <img src={bellIcon} alt="notification icon"></img>
        <div className="user_icon">
          <img
            src={userIconImage ? userIconImage : userIcon}
            alt="user icon"
            className={userIconImage ? "rounded-circle" : ""}
          />
        </div>
      </div>
    </div>
  );
};

export default Header;
