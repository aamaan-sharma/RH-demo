import React from "react";
import ToggleSwitch from "../common/toggleSwitch/toggleSwitch";
import Dropdown from "../common/dropdown/dropdown";
import "./filterSection.scss";

const stateList = [
  "Arizona",
  "Maryland",
  "California",
  "Georgia",
  "Nevada",
  "Wisconsin",
  "Texas",
  "Utah",
  "California",
  "Georgia",
  "Nevada",
  "Wisconsin",
  "Texas",
  "Utah",
];
const contractTypeList = ["RE", "DTC"];
const planList = {
  RE: ["ShieldEssential", "ShieldPlus", "ShieldComplete"],
  DTC: ["ShieldSilver", "ShieldGold", "ShieldPlatinum"],
};

const FilterSection = ({
  error,
  setError,
  selectedContract,
  setSelectedContract,
  selectedPlan,
  setSelectedPlan,
  selectedState,
  setSelectedState,
  setGptModel,
  selectedModel,
  userEmail,
}) => {
  const handleStateChange = (state) => {
    setSelectedState(state);
    setSelectedContract("Contract Type");
    setSelectedPlan("Plan");

    if (error.includes("state")) {
      setError(error.replace("state", "").trim());
    }
  };

  const handleContractChange = (contract) => {
    setSelectedContract(contract);
    setSelectedPlan("Plan");

    if (error.includes("contract")) {
      setError(error.replace("contract", "").trim());
    }
  };

  const handlePlanChange = (plan) => {
    setSelectedPlan(plan);

    if (error.includes("plan")) {
      setError(error.replace("plan", "").trim());
    }
  };

  // Determine the plan options based on the selected contract
  const filteredPlanList = planList[selectedContract] || ["Plan"];

  return (
    <div className="filter_container">
      <div className="toggle_switch_part">
        <ToggleSwitch
          setGptModel={setGptModel}
          selectedModel={selectedModel}
          userEmail={userEmail}
          isfixed={false}
        />
      </div>
      <div className="dropdown_part">
        <Dropdown
          dropdownName="State"
          selectedValue={selectedState}
          optionsList={stateList}
          highlightInput={error?.includes("state") ? true : false}
          onhandleClick={handleStateChange}
        />
        <Dropdown
          dropdownName="Contract Type"
          selectedValue={selectedContract}
          optionsList={contractTypeList}
          highlightInput={error?.includes("contract") ? true : false}
          onhandleClick={handleContractChange}
        />
        <Dropdown
          dropdownName="Plan"
          selectedValue={selectedPlan}
          optionsList={filteredPlanList}
          highlightInput={error?.includes("plan") ? true : false}
          onhandleClick={handlePlanChange}
        />
      </div>
    </div>
  );
};

export default FilterSection;
