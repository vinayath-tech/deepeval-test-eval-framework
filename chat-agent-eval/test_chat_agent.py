from deepeval.test_case import MultiTurnParams, Turn, ConversationalTestCase
from deepeval.metrics import ConversationalGEval, TurnRelevancyMetric, KnowledgeRetentionMetric, ConversationCompletenessMetric
from deepeval import evaluate
from chat_agent import chat, ORDERS, REFUND_POLICIES
from config import CHAT_JUDGE_MODEL_LOCAL, CHAT_JUDGE_MODEL_OPENAI

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
        turns=turns,
        scenario=(
        """ A ShopEasy customer asks for one order's status, then asks about 
        refund policies for clothing and for food."""
         ),
        expected_outcome=(
            f"ORD-1042 is {ORDERS['ORD-1042']['status']} with ETA {ORDERS['ORD-1042']['eta']}. "
            f"Clothing: {REFUND_POLICIES['clothing']} "
            f"Food: {REFUND_POLICIES['food']}"
        )
    )

    # Commenting out below metrics to reduce the eval execution time take by ollama model.
    # turnRelevancyMetric = TurnRelevancyMetric(threshold=0.7, model=CHAT_AGENT_MODEL)
    # knowledgeRetentionMetric = KnowledgeRetentionMetric(threshold=0.5, model=CHAT_AGENT_MODEL)
    # conversationCompletenessMetric = ConversationCompletenessMetric(threshold=0.5, model=CHAT_AGENT_MODEL)

    knowledgeRetentionMetric = KnowledgeRetentionMetric(threshold=0.5, model=CHAT_JUDGE_MODEL_LOCAL)

    # GEval metrics
    correctness = ConversationalGEval(
        name = "Correctness",
        criteria = (
            "Check every refund or order-status answer the assistant gives "
            "are factually correct."
        ),
        threshold = 0.5,
        evaluation_params= [MultiTurnParams.ROLE, MultiTurnParams.CONTENT],
        model=CHAT_JUDGE_MODEL_OPENAI
    )


    evaluate(test_cases=[test_case], metrics = [knowledgeRetentionMetric, correctness])