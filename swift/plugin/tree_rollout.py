from dataclasses import dataclass, field
import copy
from enum import Enum
from typing import List

from modelscope.preprocessors.templates.utils import Messages

class SampleStatus(Enum):
    INITIAL = "initial"
    TO_INFER = "to_infer"
    FINISH_NEXT_INFER = "finish_next_infer"
    FINISHED = "finished"

@dataclass
class DataSampleTree:
    """
        Attributes:
            tree_idx: str
                for example 0/1-2/2-3/4-0, root_node = 0, next node = 1-2 infer batch 1 and index 2 sample

    """
    tree_idx: str
    request_id: str

    messages: Messages
    logprobs: List[List[float]] = field(default_factory=list)

    response_ids: List[int] = field(default_factory=list)
    all_response_ids: List[List[int]] = field(default_factory=list)

    token_count_per_step: List[int] = field(default_factory=list)

    status: SampleStatus = SampleStatus.INITIAL

    @property
    def root_node(self):
        return int(self.tree_idx.split('/')[0])

    @property
    def depth(self):
        return len(self.tree_idx.split('/')) - 1

    def extend_response(self, response_id: List[int], response_text: str, logprobs: List[float]):
        self.all_response_ids.append(response_id)
        self.extend_response_text(response_text)
        self.extend_logprobs(logprobs)

    def extend_response_text(self, response_text: str):
        self.messages.append({
            "role": 'assistant',
            'content': response_text
        })

    def extend_logprobs(self, logprobs: List[float]):
        self.logprobs.append(logprobs)

def _repeat_list_interleave(any_list, repeat_times):
    # return [item for sublist in [[item] * repeat_times for item in any_list] for item in sublist]
    return [copy.deepcopy(item) for sublist in [[item] * repeat_times for item in any_list] for item in sublist]

def _increment_tree_idx_depth(
    samples: list[DataSampleTree],
    next_infer_step: int,
    ) -> list[DataSampleTree]:
    for infer_batch_idx, sample in enumerate(samples):
        sample.tree_idx = sample.tree_idx + "/" + f"{next_infer_step}-{infer_batch_idx}"
    return samples