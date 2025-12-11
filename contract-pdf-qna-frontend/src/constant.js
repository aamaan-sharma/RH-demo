import coverageLimitIcon from "./assets/coverage_limit.svg";
import serviceIcon from "./assets/service.svg";
import policyIcon from "./assets/policy.svg";
import secondaryIcon from "./assets/secondary_damage.svg";
import commercialUseIcon from "./assets/commercial_use.svg";
import inaccessibleLocationIcon from "./assets/inaccessible_location.svg";
import clockIcon from "./assets/clock.svg";

export const searchPrompts = [
  {
    category_title: "Coverage/Limits",
    icon: coverageLimitIcon,
    questions: [
      {
        tag: "Item Coverage",
        text: "Are the fire sprinkler systems covered?",
      },
      {
        tag: "Coverage Limit",
        text: "What is the refrigerant cost that the plan covers?",
      },
      {
        tag: "Other Limits",
        text: "I have paid $1500 for modification and there was a code violation, how much will I get reimbursed?",
      },
    ],
  },
  {
    category_title: "Service",
    icon: serviceIcon,
    questions: [
      {
        tag: "Diagnosis on weekends",
        text: "I am unavailable on the weekdays, can I get the diagnosis and repair done on Saturday or Sunday?",
      },
      {
        tag: "Workmanship Guarantee",
        text: "I got my bell fixed a few days ago but it's not working again. Do I need to pay again?",
      },
      {
        tag: "Outside Service Contractor",
        text: "I know a person who can fix things well. Can I get my appliance fixed by him instead?",
      },
    ],
  },
  {
    category_title: "Policy",
    icon: policyIcon,
    questions: [
      {
        tag: "Coverage - waiting period ",
        text: "If there is a breakdown during the waiting period can a member file a claim on that item once the waiting period ends?",
      },
      {
        tag: "Plan termination",
        text: "I’m planning on selling my house. What needs to be done?",
      },
      {
        tag: "Refund on Cancellation",
        text: "I want to cancel my agreement. How does the refund system work?",
      },
    ],
  },
];

export const inferPrompts = [
  {
    category_title: "Secondary Damage",
    icon: secondaryIcon,
    questions: [
      {
        text: "My refrigerator is leaking. I need to get it fixed. The leak has damaged the ceiling fan directly below it. I need to get the ceiling fan fixed too.",
      },
    ],
  },
  {
    category_title: "Commercial Use",
    icon: commercialUseIcon,
    questions: [
      {
        text: "The dishwasher in my basement turned kitchen for my catering service is not able to clean the utensils properly. I think the soap dispenser is not working. Is it covered?",
      },
    ],
  },
  {
    category_title: "Inaccessible Location",
    icon: inaccessibleLocationIcon,
    questions: [
      {
        text: "I think there is a leak in my bathroom but it seems to be behind my bathroom cabinet. Will the repair for this leak be covered?",
      },
    ],
  },
  {
    category_title: "Waiting period",
    icon: clockIcon,
    questions: [
      {
        text: "I purchased a plan from AHS just 5 days ago, and now I want to repair the microwave because it is creating too much noise. Can I get this repair done?",
      },
    ],
  },
];

export const mockHistoryList = [
  {
    conversationName: "What is the benifits of nutrilite liver powder",
    conversationId: "11122222",
  },
  {
    conversationName: "Give me template",
    conversationId: "11122222",
  },
];

export const mockChatList = [
  {
    entered_query:
      "My refrigerator is leaking. I need to get it fixed. The leak has damaged the ceiling fan directly below it. I need to get the ceiling fan fixed too.",
    response:
      "The repair for your leaking refrigerator is covered by the plan, provided the leak is not caused by hazardous materials, misuse, known pre-existing issues, or cosmetic damage. However, the secondary damage to the ceiling fan caused by the refrigerator leak is not covered under the plan",
  },
];
