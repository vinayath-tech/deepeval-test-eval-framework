from deepeval.test_case import MultiTurnParams, Turn, ConversationalTestCase
from deepeval.metrics import ConversationalGEval, TurnRelevancyMetric, KnowledgeRetentionMetric, ConversationCompletenessMetric
from deepeval import evaluate
from chat_agent import chat

def test_chat_agent_evaluation():
    """Test the chat agent with conversational metrics"""
    turns = []
    history = []

    for user_msg in [
        "Give me status of ORD-1042?",
        "What is the refund policy?",
        "For clothing category?",
        "What about food?"
    ]:
        reply,history,_ = chat(user_msg, history)
        turns.append(Turn(role="user", content=user_msg))
        turns.append(Turn(role="assistant", content=reply))

    test_case = ConversationalTestCase(
        turns=turns
    )

    turnRelevancyMetric = TurnRelevancyMetric(threshold=0.7)
    knowledgeRetentionMetric = KnowledgeRetentionMetric(threshold=0.5)
    conversationCompletenessMetric = ConversationCompletenessMetric(threshold=0.5)

    # GEval metrics
    correctness = ConversationalGEval(
        name = "Correctness",
        criteria = (
            "Check every refund or order-status answer the assistant gives "
            "are factually correct."
        ),
        threshold = 0.5,
        evaluation_params= [MultiTurnParams.ROLE, MultiTurnParams.CONTENT]
    )


    evaluate(test_cases=[test_case], metrics = [turnRelevancyMetric, knowledgeRetentionMetric, conversationCompletenessMetric, correctness])