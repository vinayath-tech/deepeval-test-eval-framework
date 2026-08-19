from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import observe, update_current_trace
from order_agent import support_agent as _support_agent
from deepeval.contextvars import get_current_golden

@observe(name="Support agent")
def support_agent(user_input: str) -> str:
    golden = get_current_golden()
    if golden and golden.expected_output:
        update_current_trace(expected_output = golden.expected_output)

    return _support_agent(user_input)

correctness = GEval(
    name="Correctness metric",
    criteria=("""Determine if the actual output is factually correct based on the
        expected output. Don't need to be exactly the same words. But fail the
        metric if the actual output provide incorrect information or missing data"""
    ),
    threshold=0.8,
    evaluation_params=[
        SingleTurnParams.INPUT, 
        SingleTurnParams.EXPECTED_OUTPUT, 
        SingleTurnParams.ACTUAL_OUTPUT
    ]
)

dataset = EvaluationDataset(goldens=[Golden(
    input="Where is my Order ORD-001?",
    expected_output="Order ORD-001 is shipped and is expected to arrive on 17th Aug 2026"
)])

for golden in dataset.evals_iterator(metrics=[correctness]):
    support_agent(golden.input)