import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

load_dotenv(find_dotenv())

class PromptManager:
      """Manages system prompt for different tasks"""

      SUMMARY_PROMPT = """You are an AI assistant summarizing meeting transcripts. Provide a clear and 
concise summary of the following conversation, avoiding interpretation and 
unnecessary details. Focus on the main discussion points only. Do not include 
any action items. Respond with only the summary as plain text — no headings, 
formatting, or explanations."""

      ACTION_ITEM_PROMPT="""
                Extract all action items from the following meeting transcript. Identify individual 
                and team-wide action items in the following format:

                {
                "individual_actions": {
                    "Alice": ["Task 1", "Task 2"],
                    "Bob": ["Task 1"]
                },
                "team_actions": ["Task 1", "Task 2"],
                "entities": ["Alice", "Bob"]
                }

                Only include what is explicitly mentioned. Do not infer. You must respond strictly in 
                valid JSON format — no extra text or commentary."""

      @classmethod
      def get_summary_prompt(cls) -> str:
            return cls.SUMMARY_PROMPT
      
      @classmethod
      def get_action_item_prompt(cls) -> str:
            return cls.ACTION_ITEM_PROMPT
      

class MeetingSummarizer:

      def __init__(
                  self,
                  model: str="gpt-4.1",
                  temperature: float = 0.7,
                  api_key = None

      ):
            
            """ Initialize summarizer with model configuration"""

            self.model = model
            self.temperature = temperature
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")

            # Initialize OpenAI client
            self.client=OpenAI(api_key=self.api_key)

            # Initialize prompt manager
            self.prompt_manager = PromptManager()

      def _call_api(self, system_prompt: str, user_content: str):

            """
                  Method to call Open AI
            """
            try:
                  response = self.client.chat.completions.create(
                        model=self.model,
                        temperature=self.temperature,
                        messages=[
                              {"role": "system", "content": system_prompt},
                              {"role": "user", "content": user_content}
                        ]
                  )
                  return response.choices[0].message.content.strip()
            except Exception as e:
                  raise Exception(f"OpenAI API call failed: {str(e)}")
            
      def get_summary(self, transcript: str) -> str:
            """
                  Generate summary from meeting script
            """
            if not transcript or not transcript.strip():
                  raise ValueError("Transcript cannot be empty")
            
            try:
                  prompt = self.prompt_manager.get_summary_prompt()
                  return self._call_api(prompt, transcript)
            except Exception as e:
                  return f"Could not generate summary: {str(e)}"
            
      def get_action_items(self, transcript: str) -> str:
            """
               Extract action items from meeting transcript
            """

            if not transcript or not transcript.strip():
                  raise ValueError("Transcript cannot be empty")
            
            try:
                  prompt = self.prompt_manager.get_action_item_prompt()
                  return self._call_api(prompt, transcript)
            except Exception as e:
                  return f"Could not extract action items: {str(e)}"
            
      def process(self, transcript: str) -> tuple[str, str]:
            """
                  Generate both summary and action items
            """
            summary = self.get_summary(transcript)
            action_items = self.get_action_items(transcript)
            return summary, action_items


_default_summarizer = MeetingSummarizer()

def summarize(transcript: str) -> tuple[str, str]:
      """ Legacy functions interface"""
      return _default_summarizer.process(transcript)

def get_summary(transcript: str) -> str:
      return _default_summarizer.get_summary(transcript)
        
        
def get_action_items(transcript: str) :
      return _default_summarizer.get_action_items(transcript)
        