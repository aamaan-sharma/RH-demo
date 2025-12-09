import React, { useState, useRef, useEffect } from "react";
import downArrow from "../../../assets/down_arrow.svg";
import "./dropdown.scss";

const Dropdown = ({
  dropdownName,
  selectedValue,
  optionsList,
  onhandleClick,
  highlightInput,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState(selectedValue || "");
  const searchInputRef = useRef(null);
  const dropdownRef = useRef(null);

  const toggleDropdown = () => {
    setIsOpen((prevState) => {
      if (!prevState) {
        setSearchTerm(""); // Clear search term when opening the dropdown
        setTimeout(() => {
          if (searchInputRef.current) {
            searchInputRef.current.focus(); // Focus the search input when dropdown opens
          }
        }, 0);
      }
      return !prevState;
    });
  };

  const handleSearch = (event) => {
    setSearchTerm(event.target.value);
  };

  const handleClickOutside = (event) => {
    if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
      setIsOpen(false);
    }
  };

  const filteredOptions = optionsList.filter((option) =>
    option.toLowerCase().includes(searchTerm.toLowerCase())
  );

  useEffect(() => {
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  useEffect(() => {
    if (!isOpen) {
      setSearchTerm(selectedValue);
    }
  }, [isOpen, selectedValue]);

  return (
    <div className="dropdown" ref={dropdownRef}>
      <div
        className={`header ${highlightInput ? "flash" : ""} ${
          isOpen && "selected"
        }`}
        onClick={toggleDropdown}
      >
        <input
          ref={searchInputRef}
          type="text"
          placeholder={dropdownName}
          value={isOpen ? searchTerm : selectedValue || ""}
          onChange={handleSearch}
          className={`search-bar ${selectedValue === dropdownName || !selectedValue ? 'default' : 'selected'}`}
          readOnly={!isOpen}
        />
        <span className={`arrow ${isOpen ? "up" : "down"}`}>
          <img src={downArrow} alt="Dropdown Arrow" />
        </span>
      </div>
      {isOpen && (
        <div className="menu">
          {filteredOptions.length > 0 ? (
            filteredOptions.map((option, index) => (
              <div
                key={index}
                className="item"
                onClick={() => {
                  onhandleClick(option);
                  setSearchTerm(option);
                  setIsOpen(false);
                }}
              >
                {option}
              </div>
            ))
          ) : (
            <div className="no-options">No options found</div>
          )}
        </div>
      )}
    </div>
  );
};

export default Dropdown;
