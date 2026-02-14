from typing import Optional
import os
from forecasting_tools.data_models.questions import (
    BinaryResolution, NumericResolution, DateResolution, MultipleChoiceResolution, ResolutionType
)
from forecasting_tools import (
    MetaculusQuestion, BinaryQuestion, MultipleChoiceQuestion,
    NumericQuestion, DateQuestion, MetaculusClient
)
from abc import ABC, abstractmethod

class AutoResolver(ABC):
    """
    Auto resolvers are provided a metaculus question, and aggregate research on
    - whether they have resolved or not
    - whether they have resolved conclusively or not
    - what conclusion they have reaches  
    """

    @abstractmethod
    def resolve_question(self, question: MetaculusQuestion) -> Optional[ResolutionType]:
        pass
        
class CommunityForecastResolver(AutoResolver):
    """
    Checks if the community forecast has reached a consensus. This only works on binary forecasts
    at the minute.

    This should not be used alone, as there can be extremely slim chanced theoretical questions
    (i.e. what is the chance that a meteor hits the earth in the next year) that have not resolved.

    This is also unable to determine if a question should resolve as annulled or ambiguous.
    """
    
    def __init__(self, binary_threshold: float = 1, mc_threshold: float = 1):
       self.binary_theshold = binary_threshold
       self.mc_threshold = mc_threshold

    @abstractmethod
    def resolve_question(self, question: MetaculusQuestion) -> Optional[ResolutionType]:
        # Update the question
        question = MetaculusClient.get_question_by_post_id(question.id_of_post)
        if isinstance(question, BinaryQuestion):
            return self._resolve_binary_question(question)
        else:
            return None
      

    def _resolve_binary_question(self, question: BinaryQuestion) -> Optional[BinaryResolution]:
        if question.community_prediction_at_access_time is None or type(self.binary_theshold) is not int and type(self.binary_theshold) is not float:
            return None
        if self.binary_theshold + question.community_prediction_at_access_time >= 100:
            return True
        elif question.community_prediction_at_access_time - self.binary_theshold <= 0:
            return False
        else:
            return None
    
