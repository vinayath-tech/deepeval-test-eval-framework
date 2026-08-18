from deepeval.dataset import EvaluationDataset, Golden
from deepeval.contextvars import get_current_golden
from deepeval.metrics import (
    TaskCompletionMetric, 
    ToolCorrectnessMetric, 
    StepEfficiencyMetric, 
    PromptAlignmentMetric, 
    PlanQualityMetric
)
from deepeval.test_case import ToolCall
from deepeval.tracing import observe
from deepeval.tracing.context import update_current_trace
from order_agent import support_agent as _support_agent
from deepeval.evaluate.configs import AsyncConfig

task_completion  = TaskCompletionMetric(threshold=0.7, model="gpt-4.1")
tool_correctness = ToolCorrectnessMetric()
step_efficiency = StepEfficiencyMetric(threshold=0.5, model="gpt-4.1")
prompt_alignment = PromptAlignmentMetric(
            prompt_instructions=[
                "You are a friendly customer-support agent. "
                "Keep replies short and helpful."
            ], threshold=0.5, model="gpt-4.1"
)
plan_quality = PlanQualityMetric(threshold=0.7, model="gpt-4.1")

@observe(name="support agent")
def support_agent(user_input: str) -> str:
    golden = get_current_golden()
    if golden and golden.expected_tools:
        update_current_trace(expected_tools=golden.expected_tools)

    return _support_agent(user_input)

dataset =  EvaluationDataset(goldens=[
    Golden(
        input="Where is my Order ORD-001?",
        expected_tools=[ToolCall(name="get_order_status")]
    ),
    Golden(
        input="What is the refund policy for electronics?",
        expected_tools=[ToolCall(name="get_refund_policy")]
    )
])

for golden in dataset.evals_iterator(
    metrics = [task_completion, tool_correctness, prompt_alignment, plan_quality],
    async_config=AsyncConfig(run_async=False)
):
    support_agent(golden.input)