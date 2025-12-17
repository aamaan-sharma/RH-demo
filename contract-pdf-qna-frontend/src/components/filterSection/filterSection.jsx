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
// Status is now controlled only via explicit Approve (Close Case) action in the Case view.

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
  isCallsMode,
  isCallsGenerating = false,
  // Transcript list filter (used when picking a transcript)
  transcriptStatusFilter,
  onTranscriptStatusChange,
  // Conversation status (used when viewing an existing Calls conversation)
  conversationStatus,
  onConversationStatusChange,
  isConversationActive,
}) => {
  const isCallsCaseLocked = Boolean(isCallsMode && isConversationActive);
  const isCallsFilteringLocked = Boolean(isCallsCaseLocked || (isCallsMode && isCallsGenerating));

  const LockIcon = ({ size = 14 }) => (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
      focusable="false"
    >
      <path
        d="M7 10V8a5 5 0 0 1 10 0v2"
        stroke="#6B6B6B"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <path
        d="M6 10h12a2 2 0 0 1 2 2v7a3 3 0 0 1-3 3H7a3 3 0 0 1-3-3v-7a2 2 0 0 1 2-2Z"
        stroke="#6B6B6B"
        strokeWidth="2"
        strokeLinejoin="round"
      />
    </svg>
  );

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
        />
      </div>
      <div className="dropdown_part">
        {isCallsFilteringLocked ? (
          <div
            className="locked_pills"
            title={
              isCallsCaseLocked
                ? "Locked to this case. Exit case to change filters."
                : "Locked while we extract questions. Please wait until processing completes."
            }
          >
            <div className="pill">
              <LockIcon />
              <span className="label">State</span>
              <span className="value">{selectedState}</span>
            </div>
            <div className="pill">
              <LockIcon />
              <span className="label">Contract</span>
              <span className="value">{selectedContract}</span>
            </div>
            <div className="pill">
              <LockIcon />
              <span className="label">Plan</span>
              <span className="value">{selectedPlan}</span>
            </div>
            {isCallsCaseLocked ? (
              <div className={`pill status ${conversationStatus === "inactive" ? "closed" : "open"}`}>
                <span className="label">Case</span>
                <span className="value">{conversationStatus === "inactive" ? "Closed" : "Open"}</span>
              </div>
            ) : (
              <div className="pill status open">
                <span className="label">Case</span>
                <span className="value">Processing…</span>
              </div>
            )}
          </div>
        ) : (
          <>
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
          </>
        )}
        {isCallsMode && !isConversationActive && (
          null
        )}

        {isCallsMode && isConversationActive && !isCallsCaseLocked ? null : null}
      </div>
    </div>
  );
};

export default FilterSection;
