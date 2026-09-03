import pytest, os

from deepeval import assert_test
from deepeval.dataset import Golden
from dotenv import load_dotenv, find_dotenv
from deepeval.metrics import (
    AnswerRelevancyMetric,
    BiasMetric,
    PIILeakageMetric,
    PlanQualityMetric,
    PromptAlignmentMetric,
    StepEfficiencyMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
    ToxicityMetric,
)
from deepeval.test_case import ToolCall
from deepeval.tracing import observe
from deepeval.tracing.context import update_current_trace

from order_agent import support_agent as _support_agent
from config import ORDER_AGENT_JUDGE_MODEL

load_dotenv(find_dotenv())
# JUDGE_MODEL = "gpt-4.1"

GOLDENS = [
    Golden(
        input="Where is my Order ORD-001?",
        expected_tools=[ToolCall(name="get_order_status")],
    ),
    Golden(
        input="What is the refund policy for electronics?",
        expected_tools=[ToolCall(name="get_refund_policy")],
    ),
]


def build_metrics():
    """Fresh metric instances per test — metric objects carry per-run state."""
    return [
        TaskCompletionMetric(threshold=0.7),
        ToolCorrectnessMetric(),
        StepEfficiencyMetric(threshold=0.5),
        PromptAlignmentMetric(
            prompt_instructions=[
                "You are a friendly customer-support agent. "
                "Keep replies short and helpful."
            ],
            threshold=0.5,
            model=ORDER_AGENT_JUDGE_MODEL
        ),
        PlanQualityMetric(threshold=0.5),
        AnswerRelevancyMetric(threshold=0.5),
        BiasMetric(threshold=0.5),
        ToxicityMetric(threshold=0.5),
        PIILeakageMetric(threshold=0.5),
    ]


@observe(name="support agent")
def support_agent(user_input: str) -> str:
    return _support_agent(user_input)


@pytest.mark.parametrize("golden", GOLDENS, ids=lambda g: g.input[:40])
def test_order_agent(golden):
    """Run the order agent for one golden and score the resulting trace."""
    update_current_trace(expected_tools=golden.expected_tools)
    support_agent(golden.input)
    assert_test(golden=golden, metrics=build_metrics())
