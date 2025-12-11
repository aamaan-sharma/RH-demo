// Dummy data to simulate backend responses for transcripts and Q&A.
// All API-like reads in the frontend should import from here.

export const transcripts = [
  {
    id: "t1",
    name: "meeting_transcript_2024.txt",
    updatedAt: "2024-10-21T10:30:00Z",
  },
  {
    id: "t2",
    name: "claims_call_2024-09-12.txt",
    updatedAt: "2024-09-12T16:45:00Z",
  },
  {
    id: "t3",
    name: "policy_review_session.txt",
    updatedAt: "2024-08-05T09:15:00Z",
  },
];

export const qaByTranscript = {
  t1: {
    questions: [
      {
        question: "What appliances are covered under the policy?",
        chunks: [
          "The homeowner policy covers HVAC, plumbing, electrical systems, and major kitchen appliances. Coverage includes central air systems, ductwork, furnaces, boilers, water heaters, interior electrical panels, wiring, and built-in kitchen appliances such as ovens, ranges, cooktops, dishwashers, and refrigerators. Exclusions apply to cosmetic issues, pre-existing conditions, and poorly maintained equipment.",
          "Coverage applies to sudden and accidental damage; wear and tear exclusions apply. Failures from neglect, rust, corrosion, or improper installation are excluded. Claims require timely mitigation and documentation of the incident.",
          "Optional riders may expand coverage for secondary appliances and premium components, subject to higher limits and deductibles. Review the schedule of benefits for per-item caps and aggregate annual limits before filing a claim.",
        ],
        answer:
          "Covered: HVAC, plumbing, electrical, and major kitchen appliances for sudden/accidental damage. Excludes wear and tear.",
      },
      {
        question: "Is storm-related roof damage covered?",
        chunks: [
          "Storm-related roof damage is covered, but depreciation may apply based on roof age. Claims over ten-year-old roofs are adjusted for material type and remaining useful life. Prior repairs must be disclosed.",
          "Immediate mitigation and documentation are required. Tarping, photo evidence, and contractor estimates must be submitted within 72 hours to avoid denial for secondary water ingress.",
          "Excluded: long-term leaks, rot, mold from maintenance failures, or pre-existing defects. Gutters and solar arrays are evaluated separately and may require add-on coverage.",
        ],
        answer:
          "Yes, storm roof damage is covered, with depreciation based on roof age. Mitigation and documentation are required.",
      },
      {
        question: "Are mechanical breakdowns of appliances covered?",
        chunks: [
          "Mechanical breakdowns from normal wear and tear are not covered under standard policy. Failures from age, lack of maintenance, or manufacturer defects are excluded unless an equipment protection rider is in place.",
          "Optional appliance protection plan can supplement coverage. The add-on waives wear-and-tear exclusions for listed appliances after a 30-day waiting period and requires annual maintenance logs.",
          "Service-call fees and per-claim caps apply. Replacement may be offered when repair exceeds 70% of replacement cost, subject to like-kind-and-quality rules.",
        ],
        answer:
          "No, mechanical breakdowns from normal wear/tear aren’t covered. Consider the appliance protection add-on.",
      },
    ],
    finalAnswer:
      "Policy covers sudden/accidental damage to major systems; storm roof damage is covered with depreciation; mechanical breakdowns from wear/tear are excluded unless an appliance plan is added.",
  },
  t2: {
    questions: [
      {
        question: "What is the deductible for claims?",
        chunks: [
          "The standard deductible is $1,000 across all claims. This applies per occurrence, not per item, and must be satisfied before payout.",
          "Wind and hail in designated catastrophe zones may carry separate deductibles stated as a percentage of Coverage A; review your declarations page for specifics.",
          "Higher deductibles can lower premiums but increase out-of-pocket responsibility. Deductible waivers are not available for frequent claimants within a policy term.",
        ],
        answer: "The deductible is $1,000 for all claims.",
      },
      {
        question: "Does vandalism qualify as a covered peril?",
        chunks: [
          "Vandalism is listed as a covered peril under the policy. Forced entry, malicious damage, and defacement are included when promptly reported to authorities.",
          "Coverage excludes intentional acts by insured parties or invited guests. Claims require a police report number and photos within 48 hours of discovery.",
          "Outbuildings and detached structures are covered up to the Other Structures limit; specialty items (art, collectibles) require scheduled coverage to avoid sublimits.",
        ],
        answer: "Yes, vandalism is a covered peril under this policy.",
      },
    ],
    finalAnswer:
      "Deductible is $1,000; vandalism is covered. Follow standard claims steps and document incidents promptly.",
  },
  t3: {
    questions: [
      {
        question: "Are secondary damages covered?",
        chunks: [
          "Secondary damage from an initial covered peril is typically covered if mitigated promptly. This includes water intrusion after a storm-damaged roof when tarping was performed in a timely manner.",
          "Coverage may be reduced or denied if delays in mitigation lead to mold, rot, or additional structural harm. Maintain receipts and time-stamped photos of all mitigation steps.",
          "Claims adjusters may require contractor assessments to separate primary from secondary damage; exclusions apply to pre-existing deterioration.",
        ],
        answer:
          "Secondary damage is generally covered if it stems from a covered peril and mitigation was prompt.",
      },
    ],
    finalAnswer:
      "Secondary damages from covered perils are usually covered when promptly mitigated.",
  },
};


