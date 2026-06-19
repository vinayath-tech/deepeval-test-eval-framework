import os
from conftest import MeetingSummarizer
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval import evaluate, assert_test
from deepeval.evaluate import DisplayConfig


class TestSummary:

    def transcript_loader(self) -> list[str]:

        documents_path = "./transcript-agent-eval/dataset"
        transcripts = []

        for document in os.listdir(documents_path):
            if document.endswith(".txt"):
                file_path = os.path.join(documents_path, document)
                with open(file_path, "r", encoding="utf-8") as file:
                    transcript =  file.read().strip()
                transcripts.append(transcript)
        
        return transcripts
    

    def dataset_loader(self, get_trancripts) -> EvaluationDataset:
        goldens = []
        for transcript in get_trancripts:
            golden = Golden(
                input = transcript
            )
            goldens.append(golden)

        for i, golden in enumerate(goldens):
            print(f"Golden {i}: ", golden.input[:100])

        return EvaluationDataset(goldens=goldens)
    
    def summary_concision_metric(self):
        return self.geval_metric(
             "Summary Concision",
            "Assess whether the summary is accurate & focused only on the essential points of the meeting"
        )

    def action_item_check_metric(self):
        return self.geval_metric(
            "Action item accuracy",
            "Are the action items accurate, complete and clearly reflect the key tasks mentioned in the meeting?"
        )

    def build_test_case(self, dataset, summarizer) -> tuple[list, list]:
        summary_test_cases_list = []
        action_item_test_cases_list = []

        print(f"total length of goldens is ##### {len(dataset.goldens)}")
        # Create all test cases first
        for golden in dataset.goldens:
            summary, action_items = summarizer.process(golden.input)
            summary_test_case = LLMTestCase(
                input=golden.input,
                actual_output=summary 
            )
            assert_test(test_case=summary_test_case, metrics=[self.summary_concision_metric()])

            action_item_test_case = LLMTestCase(
                input=golden.input,
                actual_output=action_items
            )
            assert_test(test_case=action_item_test_case, metrics=[self.action_item_check_metric()])

            summary_test_cases_list.append(summary_test_case)
            action_item_test_cases_list.append(action_item_test_case)
        
        # Return after processing ALL transcripts
        return summary_test_cases_list, action_item_test_cases_list

   
    def geval_metric(self, name:str, criteria:str) -> GEval:
        return GEval(
            name = name,
            criteria = criteria,
            threshold=0.7,
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
        )


    def test_eval_summarize(self):

        summarizer = MeetingSummarizer(model="gpt-4.1", temperature=0.5)
        get_trancripts = self.transcript_loader()
        dataset = self.dataset_loader(get_trancripts)

        # Execute evaluation
        self.build_test_case(dataset, summarizer) 
