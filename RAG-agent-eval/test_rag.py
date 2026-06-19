from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.dataset import EvaluationDataset
from rag_qa_agent import RAGAgent
from deepeval.synthesizer import Synthesizer
from deepeval.metrics import (
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric
)
from deepeval.metrics import GEval
from deepeval import evaluate, assert_test


class TestRag:

    def generate_dataset(self):
        synthesizer = Synthesizer()
        goldens = synthesizer.generate_goldens_from_docs(
            document_paths = ["./RAG-agent-eval/dataset/theranos_legacy.txt"]
        )
        dataset = EvaluationDataset(goldens=goldens)
        return dataset

    def build_test_case(self):
        dataset = self.generate_dataset()
        agent = RAGAgent(document_paths=["./RAG-agent-eval/dataset/theranos_legacy.txt"])

        test_cases = []
        for golden in dataset.goldens:
            retrieved_docs = agent.retrieve(golden.input)
            response = agent.generate(golden.input, retrieved_docs)
            test_case = LLMTestCase(
                input=golden.input,
                actual_output=str(response),
                expected_output=golden.expected_output,
                retrieval_context=retrieved_docs
            )
            test_cases.append(test_case)
        
        print("I am here ####################")
        print(len(test_cases))
        return test_cases

    
    def test_eval_rag(self):
        test_cases=self.build_test_case()

        relevancy = ContextualRelevancyMetric()
        recall = ContextualRecallMetric()
        precision = ContextualPrecisionMetric()

        answer_correctness = GEval(
            name="Answer Correctness",
            criteria="Evaluate if the actual output's 'answer' property is correct and complete from the input and retrieved context. If the answer is not correct or complete, reduce score",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT]
        )

        citation_accuracy = GEval(
            name="Citation Accuracy",
            criteria="Check if the Citations in the actual outpt are correct and relevant based on the Input. If the answer is not correct or complete, reduce score",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT]
        )

        # Retriever eval
        retriever_metrics = [relevancy, recall, precision]
        failures = []
        for test_case in test_cases:
            try:
                assert_test(test_case=test_case, metrics=retriever_metrics)
            except AssertionError as e:
                print(f"Test case failed: {e}")
                failures.append(str(e))
                continue

        # Generate eval
        generate_metrics = [answer_correctness, citation_accuracy]
        for test_case in test_cases:
            try:
                assert_test(test_case=test_case, metrics=generate_metrics)
            except AssertionError as e:
                print(f"Test case failed: {e}")
                failures.append(str(e))
                continue
        
        # Fail the test if any assertions failed
        if failures:
            raise AssertionError(f"{len(failures)} test case(s) failed:\n" + "\n".join(failures))
        # evaluate(test_cases=test_cases,
        #          metrics=generate_metrics
        # )

    