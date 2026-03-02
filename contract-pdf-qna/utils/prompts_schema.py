from pydantic import BaseModel, Field
from typing import Literal, List, Tuple, Annotated


class QuestionObj(BaseModel):
    question: Annotated[
        str,
        "User question extracted from transcript, prefixed with [CALL_CONTEXT: item=...; location=...; issue=...; ...] then the human-readable claim-review question.",
    ] = ""
    context: Annotated[
        str,
        "Two- to four-sentence claim-note style summary for this item, with 1–2 short verbatim evidence quotes from the transcript.",
    ] = ""
    questionType: Annotated[
        Literal["claim_review", "coverage", "eligibility", "authorization", "costs", "process"],
        "Question type: claim_review, coverage, eligibility, authorization, costs, or process.",
    ] = "process"
    userIntent: Annotated[
        str,
        "Short phrase summarizing what the user/customer is trying to determine (e.g. determine coverage for HVAC repair, reconcile authorized total vs estimate).",
    ] = ""
    tags: Annotated[
        List[str],
        "Two-word summary (word1, word2): word1 = actionable service (e.g. coverage query, repair, inspection, authorization); word2 = subject (e.g. appliance, pipeline, kitchen sink).",
    ] = ("", "")


class QuestionsObj(BaseModel):
    questionList: Annotated[
        List[QuestionObj],
        "List of extracted claim-review questions, one per distinct item/service/work scope from the transcript.",
    ] = Field(default_factory=list)