import React from "react";
import "./insights.scss";

const Insights = () => {
  return (
    <>
      <div style={{ height: "100vh", width: "100vw", overflowY: "hidden" }}>
        <iframe
          title="dashboard"
          width="100%"
          height="100%"
          src="https://lookerstudio.google.com/embed/reporting/c39b0789-f5be-4bec-ab89-b2d0798faf39/page/usFlD"
          allowFullScreen
        ></iframe>
      </div>
    </>
  );
};

export default Insights;
