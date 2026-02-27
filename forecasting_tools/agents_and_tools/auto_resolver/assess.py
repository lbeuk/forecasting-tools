from forecasting_tools.agents_and_tools.auto_resolver import AutoResolver
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from forecasting_tools.data_models.questions import QuestionBasicType, BinaryResolution, CanceledResolution, ResolutionType
from forecasting_tools.helpers.metaculus_client import MetaculusClient
from forecasting_tools import MetaculusQuestion, ApiFilter, BinaryQuestion
from dataclasses import dataclass, field


@dataclass
class QuestionAssessmentResult:
    """Detailed assessment result for a single question."""
    
    question_id: int
    question_title: str
    question_text: str
    question_url: str
    actual_resolution: ResolutionType
    predicted_resolution: Optional[ResolutionType]
    key_evidence: Optional[list[str]] = None
    outcome_category: Optional[str] = None


@dataclass
class ResolutionAssessmentReport:
    """
    Table for binary assessment (y-axis is true value, x-axis is predicted value):

          | True | False | Unresolvable | Cancelled | Unmatched
    ------|------|-------|--------------|-----------|------------
    True  | n_tp | n_fn  | n_mp         | n_ct      | n_um_true
    False | n_fp | n_tn  | n_mn         | n_cf      | n_um_false
    Cancelled | n_tc | n_fc | n_mc     | n_cc      | n_um_ca
    """

    tp: list[int] = field(default_factory=list)
    fp: list[int] = field(default_factory=list)
    fn: list[int] = field(default_factory=list)
    tn: list[int] = field(default_factory=list)
    mp: list[int] = field(default_factory=list)
    mn: list[int] = field(default_factory=list)
    ct: list[int] = field(default_factory=list)  # True actual, Cancelled predicted
    cf: list[int] = field(default_factory=list)  # False actual, Cancelled predicted
    tc: list[int] = field(default_factory=list)  # Cancelled actual, True predicted
    fc: list[int] = field(default_factory=list)  # Cancelled actual, False predicted
    mc: list[int] = field(default_factory=list)  # Cancelled actual, Unresolvable predicted
    cc: list[int] = field(default_factory=list)  # Cancelled actual, Cancelled predicted
    um_true: list[int] = field(default_factory=list)  # True actual, unmatched predicted (error/edge case)
    um_false: list[int] = field(default_factory=list)  # False actual, unmatched predicted (error/edge case)
    um_ca: list[int] = field(default_factory=list)  # Cancelled actual, unmatched predicted (error/edge case)
    question_results: dict[int, QuestionAssessmentResult] = field(default_factory=dict)

    @property
    def n_tp(self) -> int:
        return len(self.tp)

    @property
    def n_fp(self) -> int:
        return len(self.fp)

    @property
    def n_fn(self) -> int:
        return len(self.fn)

    @property
    def n_tn(self) -> int:
        return len(self.tn)

    @property
    def n_mp(self) -> int:
        return len(self.mp)

    @property
    def n_mn(self) -> int:
        return len(self.mn)
    
    @property
    def n_ct(self) -> int:
        return len(self.ct)
    
    @property
    def n_cf(self) -> int:
        return len(self.cf)
    
    @property
    def n_tc(self) -> int:
        return len(self.tc)
    
    @property
    def n_fc(self) -> int:
        return len(self.fc)
    
    @property
    def n_mc(self) -> int:
        return len(self.mc)
    
    @property
    def n_cc(self) -> int:
        return len(self.cc)
    
    @property
    def n_um_true(self) -> int:
        return len(self.um_true)
    
    @property
    def n_um_false(self) -> int:
        return len(self.um_false)
    
    @property
    def n_um_ca(self) -> int:
        return len(self.um_ca)
    
    def binary_results_table(self) -> str:
        """
        Returns a markdown table representation of the binary assessment report.

        The "Unmatched" column contains cases where the resolver returned an unexpected
        value (e.g., NotImplemented for unsupported question types, or other edge cases).
        These are logged as warnings for debugging.

        Returns:
            str: A markdown formatted confusion matrix table
        """
        return f"""\
| Actual \\ Predicted | True | False | Unresolvable | Cancelled | Unmatched |
|--------------------|------|-------|--------------|-----------|-----------|
| True               |  {str(self.n_tp).rjust(3)} |   {str(self.n_fn).rjust(3)} |          {str(self.n_mp).rjust(3)} |       {str(self.n_ct).rjust(3)} |   {str(self.n_um_true).rjust(3)} |
| False              |  {str(self.n_fp).rjust(3)} |   {str(self.n_tn).rjust(3)} |          {str(self.n_mn).rjust(3)} |       {str(self.n_cf).rjust(3)} |   {str(self.n_um_false).rjust(3)} |
| Cancelled          |  {str(self.n_tc).rjust(3)} |   {str(self.n_fc).rjust(3)} |          {str(self.n_mc).rjust(3)} |       {str(self.n_cc).rjust(3)} |   {str(self.n_um_ca).rjust(3)} |"""

    def detailed_report(self) -> str:
        """
        Returns a complete detailed markdown report including summary and per-question details.

        Returns:
            str: A complete markdown formatted assessment report
        """
        lines = []
        lines.append("# Auto Resolver Assessment Report\n")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Summary section
        lines.append("## Summary\n")
        lines.append(self.binary_results_table())
        lines.append("")

        # Results calculation
        total = self.n_tp + self.n_fp + self.n_fn + self.n_tn + self.n_mp + self.n_mn + \
                self.n_ct + self.n_cf + self.n_tc + self.n_fc + self.n_mc + self.n_cc + \
                self.n_um_true + self.n_um_false + self.n_um_ca
        correct = self.n_tp + self.n_tn + self.n_cc
        accuracy = (correct / total * 100) if total > 0 else 0

        lines.append(f"**Total Questions:** {total}")
        lines.append(f"**Correct Predictions:** {correct} ({accuracy:.1f}%)")
        lines.append("")

        # Detailed results section
        lines.append("## Detailed Results\n")
        lines.append("")

        # Sort question results by category, then by question ID
        sorted_results = sorted(
            self.question_results.values(),
            key=lambda x: (x.outcome_category or "", x.question_id)
        )

        for result in sorted_results:
            lines.append(f"### Question {result.question_id}\n")
            lines.append(f"**Title:** {result.question_title}")
            lines.append(f"**URL:** {result.question_url}")
            lines.append("")

            # Question contents
            lines.append(f"**Question Contents:**\n")
            lines.append("> " + result.question_text.replace("\n", "\n> "))
            lines.append("")

            # Resolution comparison table
            lines.append("| Output Resolution | Correct Resolution |")
            lines.append("|-------------------|--------------------|")
            actual_str = self._resolution_to_str(result.actual_resolution)
            predicted_str = self._resolution_to_str(result.predicted_resolution)
            lines.append(f"| {predicted_str} | {actual_str} |")
            lines.append("")

            # Key evidence section
            if result.key_evidence:
                lines.append("**Key Evidence:**")
                for evidence in result.key_evidence:
                    lines.append(f"- {evidence}")
                lines.append("")

            lines.append("---\n")

        return "\n".join(lines)

    def write_to_file(self, directory: str = "reports") -> Path:
        """
        Writes the detailed report to a markdown file in the specified directory.

        Args:
            directory: Directory path to write the report to (default: "reports")

        Returns:
            Path object pointing to the written file
        """
        from pathlib import Path

        # Create directory if it doesn't exist
        reports_dir = Path(directory)
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"assessment_report_{timestamp}.md"
        filepath = reports_dir / filename

        # Write report
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.detailed_report())

        logging.info(f"Assessment report written to {filepath}")
        return filepath

    @staticmethod
    def _resolution_to_str(resolution: Optional[ResolutionType]) -> str:
        """Convert a resolution to a human-readable string."""
        if resolution is None:
            return "None (Unresolvable)"
        elif isinstance(resolution, bool):
            return "TRUE" if resolution else "FALSE"
        elif isinstance(resolution, CanceledResolution):
            return str(resolution)
        else:
            return str(resolution)


    def __str__(self):
        return self.binary_results_table()

class ResolutionAssesser:
    """
    Utility for assessing how an auto resolver behaves on a set of already resolved questions.  

    Supports concurrent resolution of questions when assessing resolvers.
    """

    def __init__(self, resolver: AutoResolver, allowed_types: list[QuestionBasicType], questions: list[int | str] = [], tournaments: list[int | str] = [], max_concurrency: int = 3):
        """
        Initialize the resolution assessor.

        Args:
            resolver: AutoResolver to assess
            allowed_types: List of question types to include
            questions: Optional list of question IDs or URLs to assess
            tournaments: Optional list of tournament IDs to load
            max_concurrency: Maximum number of questions to resolve concurrently (default: 3)
        """
        self.client = MetaculusClient()
        self.questions: dict[int, MetaculusQuestion] = {}
        self.allowed_types = allowed_types
        self.resolver = resolver
        self.tournament_ids = tournaments
        self.question_ids = questions
        self._concurrency_limiter = asyncio.Semaphore(max_concurrency)


    def _insert_question(self, question: MetaculusQuestion):
        if question.actual_resolution_time is None:
            raise ValueError("Question is not yet resolved")
        if question.get_question_type() not in self.allowed_types:
            raise ValueError("Question is not of allowed type")
        if question.id_of_post is None:
            raise ValueError("Question does not have a post id")
        self.questions[question.id_of_post] = question

    def _load_question(self, question: int | str):
        loaded = None
        if type(question) is int:
            loaded = self.client.get_question_by_post_id(question)
        elif type(question) is str:
            loaded = self.client.get_question_by_url(question)
        else:
            return NotImplemented

        if loaded is None or not isinstance(loaded, MetaculusQuestion):
            raise ValueError("unable to find question")

        self._insert_question(loaded)

    def _load_questions(self, questions: list[int | str]):
        for question in questions:
            try:
                self._load_question(question)
            except ValueError as e:
                logging.warning(f"Skipping question {question}: {e}")
                continue
               
    async def _load_tournaments(self, tournament_ids: list[str | int]):
        filter = ApiFilter(
            allowed_tournaments=tournament_ids,
            allowed_statuses=["resolved"],
            allowed_types=self.allowed_types,
            group_question_mode="exclude",
            order_by="-published_time"
        )

        questions: list[MetaculusQuestion]  = await self.client.get_questions_matching_filter(filter)

        for question in questions:
            try:
                self._insert_question(question)
            except ValueError as e:
                question_id = getattr(question, 'id_of_post', 'unknown')
                logging.warning(f"Skipping question {question_id}: {e}")
                continue

    async def _load_tournament(self, tournament_id: str | int):
        await self._load_tournaments([tournament_id])

    async def _resolve_single_question(
        self, question: MetaculusQuestion, index: int
    ) -> tuple[int, Optional[ResolutionType], dict | None]:
        async with self._concurrency_limiter:
            resolution = await self.resolver.resolve_question(question)
            metadata = self.resolver.get_last_resolution_metadata()
            return (index, resolution, metadata)

    async def assess_resolver(self) -> ResolutionAssessmentReport:
        """
        Assess the resolver against the loaded questions.

        Uses concurrent execution (via asyncio.gather) to resolve multiple
        questions in parallel, controlled by a semaphore to limit concurrency.

        Returns:
            ResolutionAssessmentReport: Detailed report of resolver performance
        """
        report = ResolutionAssessmentReport()

        if self.tournament_ids:
            await self._load_tournaments(self.tournament_ids)
        if self.question_ids:
            self._load_questions(self.question_ids)

        question_items = list(self.questions.items())

        tasks = [
            self._resolve_single_question(question, idx)
            for idx, (question_id, question) in enumerate(question_items)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result, (question_id, question) in zip(results, question_items):
            if isinstance(result, Exception):
                logging.error(f"Exception resolving question {question_id}: {result}")
                continue

            index, test_resolution, metadata = result
            true_resolution = question.typed_resolution

            key_evidence = metadata.get("key_evidence") if metadata else None

            if isinstance(true_resolution, BinaryResolution):
                outcome_category = None

                match true_resolution, test_resolution:
                    # True actual resolution cases
                    case True, True:
                        report.tp.append(question_id)
                        outcome_category = "True Positive"
                    case True, False:
                        report.fn.append(question_id)
                        outcome_category = "False Negative"
                    case True, None:
                        report.mp.append(question_id)
                        outcome_category = "Missed Positive"
                    # False actual resolution cases
                    case False, True:
                        report.fp.append(question_id)
                        outcome_category = "False Positive"
                    case False, False:
                        report.tn.append(question_id)
                        outcome_category = "True Negative"
                    case False, None:
                        report.mn.append(question_id)
                        outcome_category = "Missed Negative"
                    # Cancelled actual resolution cases
                    case CanceledResolution(), True:
                        report.tc.append(question_id)
                        outcome_category = "True Incorrectly Predicted as True"
                    case CanceledResolution(), False:
                        report.fc.append(question_id)
                        outcome_category = "Cancelled Incorrectly Predicted as False"
                    case CanceledResolution(), None:
                        report.mc.append(question_id)
                        outcome_category = "Cancelled Predicted as Unresolvable"
                    case CanceledResolution(), CanceledResolution():
                        report.cc.append(question_id)
                        outcome_category = "Correct Cancel"
                    # True/False actual with Cancelled predicted
                    case True, CanceledResolution():
                        report.ct.append(question_id)
                        outcome_category = "True Incorrectly Predicted as Cancelled"
                    case False, CanceledResolution():
                        report.cf.append(question_id)
                        outcome_category = "False Incorrectly Predicted as Cancelled"
                    # Catch-all for unmatched cases (edge cases, errors, NotImplemented, etc.)
                    case _:
                        logging.warning(
                            f"Question {question_id} had unmatched prediction: "
                            f"true_resolution={true_resolution}, test_resolution={test_resolution}"
                        )
                        if true_resolution == True:
                            report.um_true.append(question_id)
                            outcome_category = "Unmatched - True"
                        elif true_resolution == False:
                            report.um_false.append(question_id)
                            outcome_category = "Unmatched - False"
                        elif isinstance(true_resolution, CanceledResolution):
                            report.um_ca.append(question_id)
                            outcome_category = "Unmatched - Cancelled"
                        else:
                            continue

                # Create detailed result
                question_result = QuestionAssessmentResult(
                    question_id=question_id,
                    question_title=question.question_text[:100] if question.question_text else "No title",
                    question_text=question.question_text or "No text available",
                    question_url=question.page_url or f"https://metaculus.com/{question_id}",
                    actual_resolution=true_resolution,
                    predicted_resolution=test_resolution,
                    key_evidence=key_evidence,
                    outcome_category=outcome_category,
                )
                report.question_results[question_id] = question_result
        return report