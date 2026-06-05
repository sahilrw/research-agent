"""Typed contracts for the research pipeline.

Every stage produces/consumes these models so we can swap implementations
and run ablations cleanly.
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class SubQuestion(BaseModel):
    id: int = Field(
        ..., 
        description="Unique incrementing ID starting at 1"
        )
    question: str = Field(
        ..., 
        description="An atomic, specific, and searchable sub-question."
        )
    depth: int = Field(
        default=0, 
        description="The iterative generation layer of this node in the research tree."
        )
    parent_id: Optional[int] = Field(
        default=None, 
        description="The ID of the sub-question that spawned this node."
        )
    confidence: Optional[float] = Field(
        default=None, 
        description="The confidence score assigned during gap analysis."
        )


class Plan(BaseModel):
    query_type: Literal["recent_events", "technical", "multi_hop", "ambiguous", "general"] = Field(
        ..., 
        description="The classified type of the incoming query which dictates the planning strategy."
    )
    needs_recency: bool = Field(
        ..., 
        description="True if answering this query accurately requires real-time information from the most recent ~12 months."
    )
    rationale: str = Field(
        ..., 
        description="A single sentence explaining why this decomposition and strategy was chosen. Invaluable for debugging."
    )
    sub_questions: List[SubQuestion] = Field(
        ..., 
        min_length=3, 
        max_length=5, 
        description="A list of 3 to 5 atomic, non-overlapping sub-questions."
    )


class Source(BaseModel):
    url: str
    title: str
    domain: str
    published: str | None = None
    snippet: str | None = None
    credibility: float | None = Field(
        default=None, 
        description="0.0-1.0 aggregate score"
        )
    credibility_components: dict[str, float] | None = None


class Claim(BaseModel):
    text: str
    source_indices: list[int] = Field(
        default_factory=list, 
        description="Indices into Report.sources")
    sub_question_id: int = Field(
        ..., 
        description="ID of the sub-question this claim answers"
        )


class GapAnalysis(BaseModel):
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="0.0-1.0 evaluation of how comprehensively the claims answer the sub-question."
    )
    gaps: List[str] = Field(
        ..., 
        description="Specific unresolved technical aspects or uncorroborated points."
        )
    needs_deepening: bool = Field(
        ..., 
        description="True if confidence is below threshold and budget remains."
        )
    follow_up_sub_questions: List[str] = Field(
        default_factory=list, 
        max_length=2, 
        description="0 to 2 atomic, search-optimized sub-questions addressing the identified gaps."
    )


class ReaderOutput(BaseModel):
    claims: List[Claim] = Field(
        ..., 
        description="List of atomic factual claims extracted from the sources."
        )


class Synthesis(BaseModel):
    summary: str = Field(
        ..., 
        description="Final cited prose answer to the user's query, with inline [N] citation markers."
    )
    used_source_indices: List[int] = Field(
        ..., 
        description="List of all unique global source indices explicitly cited inside the summary prose."
    )
    contradictions: List[str] = Field(
        default_factory=list, 
        description="A list of strings. Each item is one notable disagreement between sources, phrased atomically. If no contradictions exist, return [] (an empty list), NEVER a string explanation."
    )
    confidence_assessment: str = Field(
        ..., 
        description="Honest prose evaluation detailing the overall structural completeness or gaps in the final answer."
    )


class Report(BaseModel):
    query: str
    summary: str
    claims: List[Claim]
    sources: List[Source]
    sub_questions: List[SubQuestion]
    contradictions: List[str] = Field(default_factory=list)
    confidence_assessment: Optional[str] = None

