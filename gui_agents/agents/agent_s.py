import logging
import platform
from typing import Dict, List, Tuple, Callable, Optional

from gui_agents.agents.grounding import ACI
from gui_agents.agents.worker import Worker

logger = logging.getLogger("desktopenv.agent")


class UIAgent:
    """Base class for UI automation agents"""

    def __init__(
        self,
        worker_engine_params: Dict,
        grounding_agent: ACI,
        platform: str = platform.system().lower(),
    ):
        """Initialize UIAgent

        Args:
            worker_engine_params: Configuration parameters for the worker LLM agent
            grounding_agent: Instance of ACI class for UI interaction
            platform: Operating system platform (macos, linux, windows)
        """
        self.worker_engine_params = worker_engine_params
        self.grounding_agent = grounding_agent
        self.platform = platform

    def reset(self) -> None:
        """Reset agent state"""
        pass

    def predict(self, instruction: str, observation: Dict) -> Tuple[Dict, List[str]]:
        """Generate next action prediction

        Args:
            instruction: Natural language instruction
            observation: Current UI state observation

        Returns:
            Tuple containing agent info dictionary and list of actions
        """
        pass


class AgentS3(UIAgent):
    """Agent that uses no hierarchy for less inference time"""

    def __init__(
        self,
        worker_engine_params: Dict,
        grounding_agent: ACI,
        platform: str = platform.system().lower(),
        max_trajectory_length: int = 8,
        enable_reflection: bool = True,
        task_retrieval: Optional[Callable[[str], str]] = None,
        step_retrieval: Optional[Callable[[Dict, float], object]] = None,
        step_retrieval_threshold: float = 0.8,
        enable_verify: bool = False,
        verify_engine_params: Optional[Dict] = None,
    ):
        """Initialize a minimalist AgentS2 without hierarchy

        Args:
            worker_engine_params: Configuration parameters for the worker agent.
            grounding_agent: Instance of ACI class for UI interaction
            platform: Operating system platform (darwin, linux, windows)
            max_trajectory_length: Maximum number of image turns to keep
            enable_reflection: Creates a reflection agent to assist the worker agent
            enable_verify: Whether to enable pre-execution plan verification
            verify_engine_params: Optional separate engine params for the verify agent
        """

        super().__init__(worker_engine_params, grounding_agent, platform)
        self.max_trajectory_length = max_trajectory_length
        self.enable_reflection = enable_reflection
        self.task_retrieval = task_retrieval
        self.step_retrieval = step_retrieval
        self.step_retrieval_threshold = step_retrieval_threshold
        self.enable_verify = enable_verify
        self.verify_engine_params = verify_engine_params
        self.reset()

    def reset(self) -> None:
        """Reset agent state and initialize components"""
        self.executor = Worker(
            worker_engine_params=self.worker_engine_params,
            grounding_agent=self.grounding_agent,
            platform=self.platform,
            max_trajectory_length=self.max_trajectory_length,
            enable_reflection=self.enable_reflection,
            task_retrieval=self.task_retrieval,
            step_retrieval=self.step_retrieval,
            step_retrieval_threshold=self.step_retrieval_threshold,
            enable_verify=self.enable_verify,
            verify_engine_params=self.verify_engine_params,
        )

    def predict(self, instruction: str, observation: Dict) -> Tuple[Dict, List[str]]:
        # Initialize the three info dictionaries
        executor_info, actions = self.executor.generate_next_action(
            instruction=instruction, obs=observation
        )

        # concatenate the three info dictionaries
        info = {**{k: v for d in [executor_info or {}] for k, v in d.items()}}

        return info, actions
