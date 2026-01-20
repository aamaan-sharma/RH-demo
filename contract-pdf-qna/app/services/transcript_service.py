"""Transcript processing service."""
import json
import re
from typing import List, Dict, Any, Optional
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from app.services.llm_factory import get_llm_factory
# Import filter function - handle both old and new locations
try:
    import sys
    from pathlib import Path
    # Try importing from utils directory (old location)
    utils_path = Path(__file__).parent.parent.parent / "utils" / "transcript_filters.py"
    if utils_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("transcript_filters", utils_path)
        transcript_filters = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(transcript_filters)
        filter_relevant_customer_questions = transcript_filters.filter_relevant_customer_questions
    else:
        raise ImportError("transcript_filters not found")
except (ImportError, Exception):
    # Fallback: define inline if import fails
    def filter_relevant_customer_questions(questions):
        """Filter questions to keep only relevant customer questions."""
        if not questions:
            return []
        rep_question_patterns = [
            'can i have your', 'what\'s your', 'may i know',
            'could you please provide', 'can you tell me your',
            'what is your', 'do you have', 'are you',
            'can you confirm', 'would you like', 'how can i help',
            'thank you for calling', 'good morning', 'good afternoon', 'good evening'
        ]
        filtered = []
        for q in questions:
            question_text = (q.get('question', '') or '').strip().lower()
            context_text = (q.get('context', '') or '').lower()
            combined = f"{question_text} {context_text}"
            is_rep_question = any(pattern in combined for pattern in rep_question_patterns)
            if not is_rep_question:
                filtered.append(q)
        return filtered


class TranscriptService:
    """Service for transcript processing and question extraction."""
    
    def __init__(self, llm_factory=None):
        """Initialize transcript service.
        
        Args:
            llm_factory: Optional LLMFactory instance
        """
        self.llm_factory = llm_factory or get_llm_factory()
    
    def extract_relevant_customer_questions(
        self,
        transcript_content: str,
        llm: Optional[ChatOpenAI] = None
    ) -> List[Dict[str, Any]]:
        """Extract relevant customer questions from transcript.
        
        Args:
            transcript_content: Transcript content
            llm: Optional ChatOpenAI instance
            
        Returns:
            List of question dictionaries
        """
        if llm is None:
            llm = self.llm_factory.create_chat_llm(model="gpt-4o")
        
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
        
        extraction_prompt = ChatPromptTemplate.from_template(extraction_prompt_template)
        extraction_chain = extraction_prompt | llm | StrOutputParser()
        
        try:
            result = extraction_chain.invoke({"transcript": transcript_content})
            # Clean the result
            result = re.sub(r'```json\n?', '', result)
            result = re.sub(r'```\n?', '', result)
            result = result.strip()
            
            questions = json.loads(result)
            
            # Apply post-extraction filtering
            questions = filter_relevant_customer_questions(questions)
            
            # Add question IDs
            for idx, q in enumerate(questions):
                q["questionId"] = f"q{idx + 1}"
            
            return questions
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from LLM: {e}")
            print(f"LLM Response: {result[:500]}")
            return []
        except Exception as e:
            print(f"Error extracting relevant customer questions: {e}")
            return []
    
    def filter_relevant_customer_questions(self, questions: List[Dict]) -> List[Dict]:
        """Filter questions to keep only relevant customer questions.
        
        Args:
            questions: List of question dictionaries
            
        Returns:
            Filtered list of relevant questions
        """
        return filter_relevant_customer_questions(questions)
    
    def extract_text_from_transcript_json(self, transcript_data: Any) -> str:
        """Extract transcript text from JSON structure.
        
        Args:
            transcript_data: JSON data (dict, str, or None)
            
        Returns:
            Extracted text string
        """
        if transcript_data is None:
            return ""
        if isinstance(transcript_data, str):
            return transcript_data
        if isinstance(transcript_data, dict):
            return (
                transcript_data.get("text")
                or transcript_data.get("transcript")
                or transcript_data.get("content")
                or ""
            )
        return ""
    
    def transcript_to_chat_turns(self, transcript_text: str, transcript_data: Optional[Dict] = None) -> List[Dict[str, str]]:
        """Convert transcript to chat turns format.
        
        Args:
            transcript_text: Raw transcript text
            transcript_data: Optional structured transcript data
            
        Returns:
            List of chat turn dictionaries with 'role' and 'text' keys
        """
        turns = []
        
        # 1) Structured diarization-like shapes
        if isinstance(transcript_data, dict):
            for key in ("utterances", "segments", "turns", "dialogue", "dialog"):
                items = transcript_data.get(key)
                if isinstance(items, list) and items:
                    for it in items:
                        if not isinstance(it, dict):
                            continue
                        speaker = (
                            it.get("speaker")
                            or it.get("role")
                            or it.get("speakerLabel")
                            or it.get("participant")
                        )
                        text = it.get("text") or it.get("utterance") or it.get("content") or it.get("message")
                        text = (text or "").strip()
                        if not text:
                            continue
                        role = self._normalize_speaker_label(speaker)
                        if role == "Unknown" and re.search(r"\bthank you for calling\b|\bhow can i assist\b", text, re.I):
                            role = "CSR"
                        turns.append({"role": role, "text": text})
                    if turns:
                        return turns
        
        # 2) Regex speaker-tag parsing from plain text
        raw = (transcript_text or "").strip()
        if not raw:
            return []
        
        normalized = raw.replace("\r\n", "\n").replace("\r", "\n")
        
        speaker_pattern = re.compile(
            r"(?mi)^\s*(?:\[(?P<bracket>customer|caller|homeowner|policyholder|csr|agent|rep|representative|technician)\]|\b(?P<plain>customer|caller|homeowner|policyholder|csr|agent|rep|representative|technician)\b)\s*[:\-]\s*"
        )
        
        matches = list(speaker_pattern.finditer(normalized))
        if matches:
            for i, m in enumerate(matches):
                start = m.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(normalized)
                label = m.group("bracket") or m.group("plain") or ""
                chunk = normalized[start:end].strip()
                if not chunk:
                    continue
                role = self._normalize_speaker_label(label)
                turns.append({"role": role, "text": chunk})
            
            if turns:
                return turns
        
        # 3) Fallback single block
        return [{"role": "Unknown", "text": normalized}]
    
    def _normalize_speaker_label(self, label: Optional[str]) -> str:
        """Normalize speaker label to Customer/CSR/Unknown.
        
        Args:
            label: Speaker label string
            
        Returns:
            Normalized label: "Customer", "CSR", or "Unknown"
        """
        if not label:
            return "Unknown"
        x = str(label).strip().lower()
        
        if any(k in x for k in ["customer", "caller", "homeowner", "policyholder", "member"]):
            return "Customer"
        
        if any(k in x for k in ["csr", "agent", "rep", "representative", "support", "dispatcher", "employee"]):
            return "CSR"
        
        return "Unknown"
    
    def llm_segment_transcript(self, transcript_text: str) -> List[Dict[str, str]]:
        """Use LLM to segment transcript into chat turns.
        
        Args:
            transcript_text: Raw transcript text
            
        Returns:
            List of chat turn dictionaries
        """
        try:
            llm = self.llm_factory.create_chat_llm(model="gpt-4o-mini", temperature=0.0)
            prompt = ChatPromptTemplate.from_template(
                """
You are given a call transcript as plain text. Convert it into a chat conversation.

Rules:
- Output ONLY valid JSON (no markdown, no extra text).
- Return an array of objects: {{"role":"CSR"|"Customer","text":"..."}}
- Group contiguous lines by the same role.
- Do NOT invent content; only re-segment the provided transcript.
- Keep each "text" concise (ideally <= 240 characters) but preserve meaning.
- If unsure who spoke a line, choose the most likely role based on context.

Transcript:
{transcript}
"""
            )
            chain = prompt | llm | StrOutputParser()
            raw = (chain.invoke({"transcript": transcript_text}) or "").strip()
            # Clean JSON
            raw = re.sub(r'```json\n?', '', raw)
            raw = re.sub(r'```\n?', '', raw)
            data = json.loads(raw)
            if isinstance(data, list):
                cleaned = []
                for it in data:
                    if not isinstance(it, dict):
                        continue
                    role = it.get("role")
                    text = (it.get("text") or "").strip()
                    if role not in ("CSR", "Customer") or not text:
                        continue
                    cleaned.append({"role": role, "text": text})
                return cleaned
            return []
        except Exception as e:
            print(f"Warning: LLM transcript segmentation failed: {e}")
            return []
    
    def extract_questions_with_agent(
        self,
        transcript_content: str,
        llm: Optional[ChatOpenAI] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract relevant customer questions from transcript using an agent-based approach.
        Uses the same extraction prompt and filtering logic as extract_relevant_customer_questions()
        to ensure consistency with Search/Infer functionality.
        
        This function is specifically designed for the Calls section (/transcripts/process endpoint).
        
        Args:
            transcript_content: Transcript content
            llm: Optional ChatOpenAI instance
            
        Returns:
            List of question dictionaries
        """
        from langchain.agents import Tool, initialize_agent, AgentType
        from langchain.memory import ConversationBufferMemory
        
        if llm is None:
            llm = self.llm_factory.create_chat_llm(model="gpt-4o")
        
        # Optimized extraction prompt with 3-step process: Understand Intent → Frame Question → Extract
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
        
        # Create a tool that uses the extraction prompt
        def extract_questions_tool(transcript: str) -> str:
            """Tool to extract relevant customer questions from transcript using the standard extraction prompt."""
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
        
        # Create the transcript analysis tool
        transcript_analysis_tool = Tool(
            name="Transcript Question Extractor",
            func=extract_questions_tool,
            description=(
                "Useful for extracting relevant customer questions from customer service transcripts using a 3-step process: "
                "1) Understand user intent (what customer wants to know), "
                "2) Frame clear atomic questions from intents, "
                "3) Extract questions with context. "
                "Focuses on coverage lookup, damage/repair issues, coverage limits, and customer problems. "
                "Excludes customer service representative questions and administrative queries. "
                "Returns a JSON array with question, context, questionType, and userIntent fields."
            ),
        )
        
        tools = [transcript_analysis_tool]
        
        # System message for the agent - optimized with 3-step process
        agent_sys_msg = """
You are a claims transcript extraction supervisor.

Use the tool "Transcript Question Extractor" with the full transcript.

Your success criteria:
- Extract ONLY customer intents (explicit or implicit): needs, questions, confusion, objections, requests, decision points.
- Exclude CSR/admin questions unless the customer explicitly adopts them.
- De-duplicate repeated intents into one canonical question.
- Output MUST be ONLY a valid JSON array of objects with:
  question, context (including 1–2 evidence quotes), questionType, userIntent

Hard rule:
- If the tool output contains any non-JSON text, fix it and return ONLY the JSON array.

Return the final JSON array and nothing else.
        """
        
        # LangChain AgentExecutor expects a BaseMemory, not a ChatMessageHistory.
        # Use a simple in-process buffer memory for this one-off extraction run.
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output",
        )
        
        result_text = ""  # Initialize to avoid NameError in exception handlers
        try:
            # Initialize agent
            agent = initialize_agent(
                agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                tools=tools,
                llm=llm,
                verbose=True,
                memory=memory,
                early_stopping_method="generate",
                handle_parsing_errors=True,
                return_intermediate_steps=True,
            )
            
            # Create prompt with system message
            new_prompt = agent.agent.create_prompt(system_message=agent_sys_msg, tools=tools)
            agent.agent.llm_chain.prompt = new_prompt
            
            # Run agent with transcript
            agent_input = f"Extract relevant customer questions from this transcript:\n\n{transcript_content}"
            print(f"DEBUG: Running agent with transcript length: {len(transcript_content)} characters")
            response = agent.invoke({"input": agent_input})
            
            print(f"DEBUG: Agent response keys: {response.keys()}")
            print(f"DEBUG: Agent output: {response.get('output', '')[:200]}")
            
            # Extract the result from agent response
            result_text = response.get("output", "")
            
            # If agent used the tool, extract from intermediate steps
            if "intermediate_steps" in response and response["intermediate_steps"]:
                print(f"DEBUG: Found {len(response['intermediate_steps'])} intermediate steps")
                # Get the last tool result
                for idx, step in enumerate(reversed(response["intermediate_steps"])):
                    print(f"DEBUG: Step {idx}: {type(step)}, length: {len(step) if isinstance(step, (list, tuple)) else 'N/A'}")
                    if len(step) > 1 and isinstance(step[1], str):
                        result_text = step[1]
                        print(f"DEBUG: Found tool result in step {idx}: {result_text[:200]}")
                        break
            
            # Clean the result - remove markdown code blocks if present
            result_text = re.sub(r'```json\n?', '', result_text)
            result_text = re.sub(r'```\n?', '', result_text)
            result_text = result_text.strip()
            
            print(f"DEBUG: Cleaned result text length: {len(result_text)}")
            print(f"DEBUG: Cleaned result text (first 500 chars): {result_text[:500]}")
            
            # Parse JSON
            try:
                questions = json.loads(result_text)
                print(f"DEBUG: Successfully parsed {len(questions)} questions from agent")
            except json.JSONDecodeError as json_err:
                print(f"DEBUG: JSON decode error: {json_err}")
                print(f"DEBUG: Attempting to extract JSON from text...")
                # Try to extract JSON array from the text
                json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
                if json_match:
                    try:
                        questions = json.loads(json_match.group())
                        print(f"DEBUG: Extracted JSON array with {len(questions)} questions")
                    except:
                        print(f"DEBUG: Failed to parse extracted JSON")
                        raise json_err
                else:
                    raise json_err
            
            # Validate questions is a list
            if not isinstance(questions, list):
                print(f"DEBUG: Questions is not a list, got {type(questions)}")
                questions = []
            
            # Apply post-extraction filtering using existing function (same as Search/Infer)
            print(f"DEBUG: Before filtering: {len(questions)} questions")
            questions = self.filter_relevant_customer_questions(questions)
            print(f"DEBUG: After filtering: {len(questions)} questions")
            
            # If no questions after agent extraction, try direct extraction as fallback
            if not questions or len(questions) == 0:
                print(f"DEBUG: Agent extraction returned no questions, trying direct extraction method...")
                try:
                    direct_questions = self.extract_relevant_customer_questions(transcript_content, llm)
                    if direct_questions and len(direct_questions) > 0:
                        print(f"DEBUG: Direct extraction found {len(direct_questions)} questions")
                        return direct_questions
                    else:
                        print(f"DEBUG: Direct extraction also returned no questions")
                except Exception as fallback_err:
                    print(f"DEBUG: Direct extraction fallback failed: {fallback_err}")
            
            # Add question IDs
            for idx, q in enumerate(questions):
                if isinstance(q, dict):
                    q["questionId"] = f"q{idx + 1}"
            
            return questions
            
        except json.JSONDecodeError as e:
            print(f"ERROR: JSON parsing failed in agent extraction: {e}")
            print(f"ERROR: Result text (first 1000 chars): {result_text[:1000] if result_text else 'N/A'}")
            print(f"ERROR: Falling back to direct extraction method...")
            # Fallback to direct extraction if agent fails
            try:
                return self.extract_relevant_customer_questions(transcript_content, llm)
            except Exception as fallback_err:
                print(f"ERROR: Fallback extraction also failed: {fallback_err}")
                return []
        except Exception as e:
            print(f"ERROR: Exception in agent extraction: {e}")
            import traceback
            traceback.print_exc()
            print(f"ERROR: Falling back to direct extraction method...")
            # Fallback to direct extraction if agent fails
            try:
                return self.extract_relevant_customer_questions(transcript_content, llm)
            except Exception as fallback_err:
                print(f"ERROR: Fallback extraction also failed: {fallback_err}")
                return []
