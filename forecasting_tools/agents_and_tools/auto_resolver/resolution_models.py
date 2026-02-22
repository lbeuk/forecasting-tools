"""Pydantic models for structured resolution output parsing.

This module defines the data models used to parse the output from the resolver
agent into typed resolution decisions.
"""

from pydantic import BaseModel, Field
from typing import Literal


class BinaryResolutionResult(BaseModel):
    """Structured output for binary question resolution.

    This model is used to parse the final output from the resolver agent
    into a typed resolution decision with supporting evidence and reasoning.

    Attributes:
        resolution_status: The final resolution determination (TRUE, FALSE,
            AMBIGUOUS, ANNULLED, or NOT_YET_RESOLVABLE)
        reasoning: A 2-4 sentence explanation of why this resolution was chosen
        key_evidence: A list of 3-5 key pieces of evidence supporting this resolution
    """

    resolution_status: Literal[
        "TRUE", "FALSE", "AMBIGUOUS", "ANNULLED", "NOT_YET_RESOLVABLE"
    ] = Field(description="The final resolution determination")

    reasoning: str = Field(
        description="2-4 sentence explanation of why this resolution was chosen"
    )

    key_evidence: list[str] = Field(
        description="3-5 key pieces of evidence supporting this resolution",
        min_length=3,
        max_length=5,
    )
