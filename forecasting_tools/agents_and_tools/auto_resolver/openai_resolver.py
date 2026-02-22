"""Multi-agent question resolver using OpenAI Agents SDK.

This module implements a multi-agent architecture for resolving Metaculus
forecasting questions. It uses a pluggable researcher pattern with a default
Perplexity-based implementation.

Architecture:
    1. Orchestrator (minimal) coordinates handoffs between agents
    2. Researcher agent performs multiple strategic searches
    3. Resolver agent analyzes research and determines resolution
    4. Structured output parsing converts to typed resolution
"""

import logging
from typing import Optional, Callable

from forecasting_tools.data_models.questions import (
    ResolutionType,
    CanceledResolution,
    BinaryResolution,
)
from forecasting_tools import MetaculusQuestion, BinaryQuestion
from forecasting_tools.agents_and_tools.auto_resolver import AutoResolver
from forecasting_tools.agents_and_tools.auto_resolver.resolution_models import (
    BinaryResolutionResult,
)
from forecasting_tools.agents_and_tools.minor_tools import (
    perplexity_reasoning_pro_search,
)
from forecasting_tools.ai_models.agent_wrappers import (
    AgentRunner,
    AgentSdkLlm,
    AiAgent,
)
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.helpers.structure_output import structure_output
from forecasting_tools.util.misc import clean_indents

logger = logging.getLogger(__name__)


class StructuredOutputParsingError(Exception):
    """Raised when structured output parsing fails.

    This exception preserves the raw unparsed output for debugging purposes.

    Attributes:
        raw_output: The unparsed output that failed parsing
        original_error: The original exception that caused the failure
    """

    def __init__(self, raw_output: str, original_error: Exception):
        self.raw_output = raw_output
        self.original_error = original_error
        super().__init__(
            f"Failed to parse structured output: {original_error}\n"
            f"Raw output available in exception.raw_output"
        )


class OpenAIResolver(AutoResolver):
    """Multi-agent resolver using OpenAI Agents SDK.

    This resolver uses a multi-agent architecture to resolve Metaculus questions:
    1. A minimal orchestrator coordinates the workflow
    2. A researcher agent performs multiple Perplexity searches
    3. A resolver agent analyzes the research and determines the resolution
    4. Structured output parsing converts the result to a typed resolution

    The researcher agent is pluggable, allowing for custom implementations.

    Attributes:
        model: LLM model identifier for the agents
        researcher_agent_factory: Factory function to create researcher agents
        structure_output_model: Model used for structured output parsing
        timeout: Timeout for LLM calls in seconds
    """

    def __init__(
        self,
        model: str = "openrouter/anthropic/claude-sonnet-4",
        researcher_agent_factory: Optional[Callable[[BinaryQuestion], AiAgent]] = None,
        structure_output_model: Optional[GeneralLlm] = None,
        timeout: int = 480,
    ):
        """Initialize OpenAI Resolver.

        Args:
            model: LLM model for agents (default: claude-sonnet-4)
            researcher_agent_factory: Optional factory function that takes a
                BinaryQuestion and returns a custom researcher AiAgent.
                If None, uses default Perplexity-based researcher.
            structure_output_model: Model for structured output parsing.
                If None, uses gpt-5.2 mini with low temperature.
            timeout: Timeout for LLM calls in seconds
        """
        self.model = model
        self.researcher_agent_factory = (
            researcher_agent_factory or self._create_default_researcher
        )
        self.structure_output_model = structure_output_model or GeneralLlm(
            "openrouter/openai/gpt-5.2",
            temperature=0.2,
            timeout=timeout,
        )
        self.timeout = timeout
        logger.info(
            f"Initialized OpenAIResolver with model={model}, timeout={timeout}s"
        )

    async def resolve_question(
        self, question: MetaculusQuestion
    ) -> Optional[ResolutionType]:
        """Resolve a Metaculus question.

        Args:
            question: The question to resolve

        Returns:
            Typed resolution or None if not yet resolvable

        Raises:
            StructuredOutputParsingError: If output parsing fails (includes raw output)
            Other exceptions: Propagated from agent execution
        """
        logger.info(
            f"Starting resolution for question {question.id_of_post}: "
            f"{question.question_text[:100]}..."
        )

        if isinstance(question, BinaryQuestion):
            return await self._resolve_binary(question)
        else:
            logger.warning(f"Question type {type(question)} not yet supported")
            return NotImplemented

    async def _resolve_binary(
        self, question: BinaryQuestion
    ) -> Optional[BinaryResolution]:
        """Resolve a binary question using multi-agent workflow.

        Workflow:
        1. Create researcher, resolver, and orchestrator agents
        2. Run orchestrator (which coordinates handoffs)
        3. Parse final output with structured output
        4. Convert to typed resolution

        Args:
            question: Binary question to resolve

        Returns:
            BinaryResolution (True/False/AMBIGUOUS/ANNULLED) or None

        Raises:
            StructuredOutputParsingError: If parsing fails
        """
        logger.info(f"Creating agent workflow for binary question {question.id_of_post}")

        # Create agents
        researcher = self.researcher_agent_factory(question)
        resolver = self._create_resolver_agent(question)
        orchestrator = self._create_orchestrator_agent(researcher, resolver)

        logger.info("Running orchestrator agent (max_turns=10)")

        # Run the workflow (non-streaming)
        result = await AgentRunner.run(
            orchestrator, "Please begin the resolution process.", max_turns=10
        )

        logger.info(
            f"Agent workflow completed. Final output length: "
            f"{len(result.final_output)} chars"
        )
        logger.debug(f"Final output preview: {result.final_output[:200]}...")

        # Parse structured output with error handling
        try:
            resolution_result = await structure_output(
                result.final_output,
                BinaryResolutionResult,
                model=self.structure_output_model,
            )
            logger.info(
                f"Successfully parsed resolution: {resolution_result.resolution_status}"
            )
        except Exception as e:
            logger.error(f"Failed to parse structured output: {e}", exc_info=True)
            raise StructuredOutputParsingError(
                raw_output=result.final_output, original_error=e
            ) from e

        # Convert to typed resolution
        typed_resolution = self._convert_to_binary_resolution(resolution_result)
        logger.info(f"Final resolution: {typed_resolution}")

        return typed_resolution

    def _create_default_researcher(self, question: BinaryQuestion) -> AiAgent:
        """Create default Perplexity-based researcher agent.

        This agent performs multiple strategic searches to gather
        comprehensive information about the question's resolution status.

        Args:
            question: The question to research

        Returns:
            Configured researcher AiAgent
        """
        logger.debug("Creating default Perplexity-based researcher agent")

        instructions = self._build_researcher_instructions(question)

        return AiAgent(
            name="Resolution Researcher",
            instructions=instructions,
            model=AgentSdkLlm(model=self.model),
            tools=[perplexity_reasoning_pro_search],
            handoffs=["resolver"],
        )

    def _create_resolver_agent(self, question: BinaryQuestion) -> AiAgent:
        """Create resolver agent that determines final resolution.

        This agent receives research from the researcher agent and
        makes the final resolution determination.

        Args:
            question: The question being resolved

        Returns:
            Configured resolver AiAgent
        """
        logger.debug("Creating resolver agent")

        instructions = self._build_resolver_instructions(question)

        return AiAgent(
            name="resolver",
            instructions=instructions,
            model=AgentSdkLlm(model=self.model),
            tools=[],  # No tools - only analyzes research
            handoffs=[],  # Terminal agent
        )

    def _create_orchestrator_agent(
        self, researcher: AiAgent, resolver: AiAgent
    ) -> AiAgent:
        """Create minimal orchestrator that enables handoffs.

        This is a simple coordinator that connects the researcher
        and resolver agents.

        Args:
            researcher: The researcher agent
            resolver: The resolver agent

        Returns:
            Minimal orchestrator AiAgent
        """
        logger.debug("Creating minimal orchestrator agent")

        instructions = clean_indents(
            """
            You are coordinating a question resolution process.

            Your task is simple:
            1. Hand off to the Resolution Researcher to gather information
            2. The researcher will hand off to the resolver when ready
            3. The resolver will provide the final resolution

            Begin by handing off to the researcher.
            """
        )

        return AiAgent(
            name="Resolution Orchestrator",
            instructions=instructions,
            model=AgentSdkLlm(model=self.model),
            tools=[],
            handoffs=[researcher, resolver],
        )

    def _build_researcher_instructions(self, question: BinaryQuestion) -> str:
        """Build detailed instructions for the researcher agent.

        Args:
            question: The question being researched

        Returns:
            Formatted instruction string
        """
        logger.debug("Building researcher instructions")

        return clean_indents(
            f"""
            # Your Role

            You are a research assistant gathering information to resolve a forecasting question.

            # The Question

            {question.give_question_details_as_markdown()}

            # Your Task

            Perform multiple strategic Perplexity searches to thoroughly investigate:

            1. **Current Status**: What is the current state of affairs related to this question?
            2. **Resolution Criteria**: Have the resolution criteria been met?
            3. **Timeline Check**: Consider the scheduled resolution date and current date
            4. **Verification**: Cross-check information from multiple sources
            5. **Edge Cases**: Look for any ambiguities, disputes, or complications

            # Search Strategy Guidelines

            - Run 3-5 searches total (don't overdo it)
            - Run searches in parallel when they're independent
            - Use follow-up searches based on initial findings
            - Focus on authoritative and recent sources
            - Note any contradictions or uncertainties you find
            - Pay special attention to dates and timelines

            # Example Search Sequence

            1. Broad search: "Current status of [topic] as of [current date]"
            2. Specific search: "Has [specific criterion] occurred?"
            3. Verification: "Latest news about [topic]"
            4. (Optional) Follow-up based on findings

            # Important Reminders

            - Be thorough but efficient
            - Document your findings clearly
            - Note the sources and dates of information
            - If you find conflicting information, document both sides
            - When ready, hand off your research to the resolver

            # Handoff

            When you've gathered sufficient information, hand off to the resolver
            with a comprehensive summary of your research findings.
            """
        )

    def _build_resolver_instructions(self, question: BinaryQuestion) -> str:
        """Build detailed instructions for the resolver agent.

        Args:
            question: The question being resolved

        Returns:
            Formatted instruction string
        """
        logger.debug("Building resolver instructions")

        return clean_indents(
            f"""
            # Your Role

            You are a resolution analyst determining the final resolution status
            of a forecasting question based on research provided to you.

            # The Question

            {question.give_question_details_as_markdown()}

            # Resolution Options

            You must determine one of the following resolutions:

            ## TRUE
            - Resolution criteria have been definitively met
            - The outcome is YES/positive
            - There is strong evidence supporting this

            ## FALSE
            - Resolution criteria have been definitively met
            - The outcome is NO/negative
            - There is strong evidence supporting this

            ## AMBIGUOUS
            - The resolution criteria occurred
            - BUT the outcome is unclear or disputed
            - Multiple interpretations are reasonable
            - Example: A law passed but its scope is unclear

            ## ANNULLED
            - A fundamental assumption of the question is false
            - The question itself is invalid or malformed
            - Example: Question asks about a company that never existed

            ## NOT_YET_RESOLVABLE
            - Insufficient information currently available
            - OR the resolution date/event hasn't occurred yet
            - OR you cannot confidently determine the resolution
            - **BE CONSERVATIVE: Default to this when uncertain**

            # Analysis Guidelines

            1. **Review the research** provided by the researcher carefully
            2. **Check the timeline**: Has the scheduled resolution date passed?
            3. **Assess the evidence**: Is it strong enough for a definitive resolution?
            4. **Consider ambiguity**: Is the outcome clear or disputed?
            5. **Be conservative**: If uncertain, return NOT_YET_RESOLVABLE

            # Critical Distinctions

            **AMBIGUOUS vs ANNULLED:**
            - AMBIGUOUS: Question is valid, but answer is unclear
            - ANNULLED: Question itself is invalid/malformed

            **FALSE vs NOT_YET_RESOLVABLE:**
            - FALSE: Definitively did NOT happen
            - NOT_YET_RESOLVABLE: Might still happen or unclear if it happened

            # Output Format

            Provide your analysis in the following format:

            **Resolution Status**: [Your chosen status]

            **Reasoning**: [2-4 sentences explaining your decision]

            **Key Evidence**:
            - [Evidence point 1]
            - [Evidence point 2]
            - [Evidence point 3]
            - [Evidence point 4 - optional]
            - [Evidence point 5 - optional]

            # Important

            - Be thorough in your reasoning
            - Cite specific information from the research
            - Acknowledge uncertainties when present
            - Your output will be parsed programmatically, so follow the format exactly
            """
        )

    def _convert_to_binary_resolution(
        self, result: BinaryResolutionResult
    ) -> Optional[BinaryResolution]:
        """Convert structured result to typed binary resolution.

        Args:
            result: Parsed resolution result

        Returns:
            Typed BinaryResolution or None

        Raises:
            ValueError: If resolution status is unexpected
        """
        logger.debug(f"Converting result status: {result.resolution_status}")

        match result.resolution_status:
            case "TRUE":
                return True
            case "FALSE":
                return False
            case "AMBIGUOUS":
                return CanceledResolution.AMBIGUOUS
            case "ANNULLED":
                return CanceledResolution.ANNULLED
            case "NOT_YET_RESOLVABLE":
                return None
            case _:
                raise ValueError(
                    f"Unexpected resolution status: {result.resolution_status}"
                )


async def main():
    """Test harness for OpenAI Resolver.

    Runs the resolver against 10 random binary questions from the AIB Fall 2025 tournament
    and outputs a confusion matrix comparing predicted vs actual resolutions.
    """
    from forecasting_tools.agents_and_tools.auto_resolver.assess import (
        ResolutionAssesser,
    )
    from forecasting_tools import MetaculusClient, ApiFilter
    from dotenv import load_dotenv
    import random

    load_dotenv()

    logger.info("Starting OpenAI Resolver assessment")

    # Fetch all resolved binary questions from AIB Fall 2025
    client = MetaculusClient()
    filter = ApiFilter(
        allowed_tournaments=[MetaculusClient.AIB_FALL_2025_ID],
        allowed_statuses=["resolved"],
        allowed_types=["binary"],
        group_question_mode="exclude",
        order_by="-published_time"
    )
    
    logger.info("Fetching resolved binary questions from AIB Fall 2025...")
    all_questions = await client.get_questions_matching_filter(filter)
    
    # Randomly sample 10 questions
    sample_size = min(20, len(all_questions))
    sampled_questions = random.sample(all_questions, sample_size)
    question_ids = [q.id_of_post for q in sampled_questions if q.id_of_post is not None]
    
    logger.info(f"Selected {len(question_ids)} random questions for assessment")

    # Create resolver
    resolver = OpenAIResolver()

    # Create assessor with specific question IDs
    assesser = ResolutionAssesser(
        resolver, allowed_types=["binary"], questions=question_ids
    )

    logger.info(f"Running assessment on {len(question_ids)} questions")

    # Run assessment
    report = await assesser.assess_resolver()

    # Print results
    print("\n" + "=" * 60)
    print("OpenAI Resolver Assessment Results")
    print(f"Tested on {len(question_ids)} random questions from AIB Fall 2025")
    print("=" * 60)
    print(report)
    print("=" * 60)

    logger.info("Assessment complete")


if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(main())
