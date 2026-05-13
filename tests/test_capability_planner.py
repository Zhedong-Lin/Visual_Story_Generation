from anime_pipeline_graph.domain.enums import TaskType
from anime_pipeline_graph.domain.models import TaskSpec
from anime_pipeline_graph.planner.capability_planner import CapabilityPlanner
from anime_pipeline_graph.providers.mock_qwen_client import MockQwenClient


def test_capability_planner_returns_plan():
    spec = TaskSpec(task_id="t1", task_type=TaskType.SINGLE_IMAGE)
    plan = CapabilityPlanner(MockQwenClient()).plan(spec)
    assert hasattr(plan, "identity_preservation")
