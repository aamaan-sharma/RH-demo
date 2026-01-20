"""Transcript Question Extractor tool for LangChain agents."""
import re
from typing import Any
from langchain.agents import Tool
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


def create_transcript_extractor_tool(llm: ChatOpenAI) -> Tool:
    """Create Transcript Question Extractor tool for agent.
    
    Args:
        llm: ChatOpenAI instance
        
    Returns:
        Tool instance for transcript extraction
    """
    extraction_prompt_template = """
You are an expert at analyzing customer service transcripts. Take the customer's perspective to understand what they need, then break that need into clear questions required to fully resolve it. Use a structured 3-step process, and infer questions even when none are explicitly asked (including when described by a technician/representative).

STEP 1: UNDERSTAND USER INTENT
- Put yourself in the customer's shoes; aim for complete customer satisfaction.
- Accept signals from any speaker (customer, technician, representative) describing the customer's issue.
- Include implicit clarifications needed to resolve the situation end-to-end (item, location, cause, limits, costs, approvals).

Focus on customer statements that express:
- Intent to understand coverage (e.g., "I want to know if...", "Is this covered?", "Will you repair...")
- Intent to understand problems (e.g., "My appliance is...", "There's a leak...", "The damage is...")
- Intent to understand policies (e.g., "What's the limit?", "How much does it cost?", "What's included?")

EXCLUDE:
- Customer service representative questions (e.g., "Can I have your name?", "What's your position?", "May I know...", "How can I help you?")
- Administrative questions
- Greetings or pleasantries
- Questions not related to coverage/repair/damage/contract/policy

STEP 2: FRAME THE QUESTIONS
For each identified customer intent, frame it as a clear, atomic question that can be answered independently. Frame questions in a way that:
- Captures the customer's actual concern or problem (in their voice)
- Is specific and answerable from contract knowledge base
- Focuses on coverage, damage, repair, policy, and any clarifiers needed to resolve the issue
- If no explicit questions are present, infer and create them. Ensure at least one question per described issue.

HARD REQUIREMENTS (Calls mode):
- Do NOT write generic questions like "Is it covered?", "Is this covered?", "Is that covered?" or "Is it covered or not?".
- Every question must be CUSTOMER-SPECIFIC: explicitly mention the appliance/system and the specific issue/service (symptom/part/service).
- Avoid vague pronouns ("it/this/that") unless you immediately clarify the appliance/issue in the same sentence.
- Generate a compact but complete set of questions that covers WH-style checks as QUESTIONS when implied by the case:
  - What failed / what service is needed
  - Where (location / affected area / on/off premises when relevant)
  - When (timing, waiting period, recent repair when relevant)
  - Why (suspected cause, secondary damage, misuse/commercial use when relevant)
  - How (repair vs replace, diagnostics, service call/trade call, limits/fees)
- Keep questions clean and professional; no filler, no disclaimers.

Question types to frame:
1. Coverage questions (contextual): "Does my plan cover diagnosing/repairing/replacing [appliance/part] for [specific failure mode]?"
2. Damage/repair questions (contextual): "Does the plan cover [specific repair/service] for [specific damage/failure] and under what limits/fees?"
3. Policy/limit questions: "What is the [specific limit/policy] for [item]?"
4. Problem statements: Convert customer problems into questions that include appliance + failure mode + requested service.
5. Clarifying questions that help resolve the customer's need (e.g., specifics about the item, location, cause, or limits that determine coverage)

STEP 3: EXTRACT AND RETRIEVE
Extract the framed questions with proper context and question type classification.

Transcript:
{transcript}

Follow this 3-step process:
1. Identify customer intents (what they want to know/understand)
2. Frame each intent as a clear, atomic question
3. Extract and return the questions

Return ONLY a JSON array of relevant customer questions in this format:
[
  {{
    "question": "Does my plan cover diagnosing and repairing my water heater tank leak described in the transcript, including any covered parts/labor and applicable fees?",
    "context": "Customer mentioned their water heater tank is leaking and causing floor damage; customer wants to know coverage for the leak and related service",
    "questionType": "coverage",
    "userIntent": "Customer wants to understand if the water heater leak they're experiencing is covered by their plan"
  }}
]

IMPORTANT:
- Extract only questions that reflect customer intent (not rep questions)
- Frame questions clearly and specifically
- Include userIntent field to show what the customer is trying to understand
- Return only valid JSON, no additional text
- If no relevant customer questions are found, return an empty array []
"""
    
    def extract_questions_tool(transcript: str) -> str:
        """Tool to extract relevant customer questions from transcript."""
        extraction_prompt = ChatPromptTemplate.from_template(extraction_prompt_template)
        extraction_chain = extraction_prompt | llm | StrOutputParser()
        
        try:
            result = extraction_chain.invoke({"transcript": transcript})
            # Clean the result - remove markdown code blocks if present
            result = re.sub(r'```json\n?', '', result)
            result = re.sub(r'```\n?', '', result)
            result = result.strip()
            return result
        except Exception as e:
            print(f"Error in extraction tool: {e}")
            return "[]"
    
    return Tool(
        name="Transcript Question Extractor",
        func=extract_questions_tool,
        description=(
            "Useful for extracting relevant customer questions from a live insurance support call. "
            "Uses a 3-step process: 1) Understand user intent (what customer wants to know), "
            "2) Frame clear atomic questions from intents, 3) Extract questions with context. "
            "Focuses on coverage lookup, damage/repair issues, coverage limits, and customer problems. "
            "Excludes customer service representative questions and administrative queries. "
            "Returns a JSON array with question, context, questionType, and userIntent fields."
        ),
    )
