from forecasting_tools.agents_and_tools.auto_resolver import AutoResolver
import asyncio
from forecasting_tools.data_models.questions import QuestionBasicType, BinaryResolution, CanceledResolution
from forecasting_tools.helpers.metaculus_client import MetaculusClient
from forecasting_tools import MetaculusQuestion, ApiFilter, BinaryQuestion
from dataclasses import dataclass, field

@dataclass
class ResolutionAssessmentReport:
    """
    Table for binary assessment (y-axis is true value, x-axis is predicted value):
    
          | True | False | Unresolvable | Cancelled
    ------|------|-------|--------------|----------
    True  | n_tp | n_fn  | n_mp         | n_ct
    False | n_fp | n_tn  | n_mn         | n_cf
    Cancelled | n_tc | n_fc | n_mc     | n_cc
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
    
    def binary_results_table(self) -> str:
        """
        Returns a markdown table representation of the binary assessment report.
        
        Returns:
            str: A markdown formatted confusion matrix table
        """
        return f"""\
| Actual \\ Predicted | True | False | Unresolvable | Cancelled |
|--------------------|------|-------|--------------|-----------|
| True               |  {str(self.n_tp).rjust(3)} |   {str(self.n_fn).rjust(3)} |          {str(self.n_mp).rjust(3)} |       {str(self.n_ct).rjust(3)} |
| False              |  {str(self.n_fp).rjust(3)} |   {str(self.n_tn).rjust(3)} |          {str(self.n_mn).rjust(3)} |       {str(self.n_cf).rjust(3)} |
| Cancelled          |  {str(self.n_tc).rjust(3)} |   {str(self.n_fc).rjust(3)} |          {str(self.n_mc).rjust(3)} |       {str(self.n_cc).rjust(3)} |"""


    def __str__(self):
        return self.binary_results_table()

class ResolutionAssesser:
    """
    Utility for assessing how an auto resolver behaves on a set of already resolved questions.  
    """

    def __init__(self, resolver: AutoResolver, allowed_types: list[QuestionBasicType], questions: list[int | str] = [], tournaments: list[int | str] = []):
        self.client = MetaculusClient()
        self.questions: dict[int, MetaculusQuestion] = {}
        self.allowed_types = allowed_types
        self.resolver = resolver
        self.tournament_ids = tournaments
        self.question_ids = questions


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
            except ValueError:
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
            except ValueError:
                continue

    async def _load_tournament(self, tournament_id: str | int):
        await self._load_tournaments([tournament_id])

    async def assess_resolver(self) -> ResolutionAssessmentReport:
        # Load tournaments if not already loaded
        if self.tournament_ids:
            await self._load_tournaments(self.tournament_ids)
        if self.question_ids:
            self._load_questions(self.question_ids)
        report = ResolutionAssessmentReport()
        for question_id, question in self.questions.items():
            true_resolution = question.typed_resolution
            test_resolution = await self.resolver.resolve_question(question)
            if isinstance(true_resolution, BinaryResolution):
                # Check if test_resolution is CanceledResolution
                test_is_cancelled = isinstance(test_resolution, CanceledResolution)
                true_is_cancelled = isinstance(true_resolution, CanceledResolution)
                
                match true_resolution, test_resolution:
                    # True actual resolution cases
                    case True, True:
                        report.tp.append(question_id)
                    case True, False:
                        report.fn.append(question_id)
                    case True, None:
                        report.mp.append(question_id)
                    # False actual resolution cases
                    case False, True:
                        report.fp.append(question_id)
                    case False, False:
                        report.tn.append(question_id)
                    case False, None:
                        report.mn.append(question_id)
                    # Cancelled actual resolution cases
                    case CanceledResolution(), True:
                        report.tc.append(question_id)
                    case CanceledResolution(), False:
                        report.fc.append(question_id)
                    case CanceledResolution(), None:
                        report.mc.append(question_id)
                    case CanceledResolution(), CanceledResolution():
                        report.cc.append(question_id)
                    # True/False actual with Cancelled predicted
                    case True, CanceledResolution():
                        report.ct.append(question_id)
                    case False, CanceledResolution():
                        report.cf.append(question_id)
        return report
