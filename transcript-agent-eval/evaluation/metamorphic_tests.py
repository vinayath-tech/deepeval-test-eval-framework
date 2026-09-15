"""
Chapter 2 metamorphic tests.

A metamorphic test changes the input in a way that should not
materially change the important properties of the AI output.

For this transcript summarizer:

Original transcript
        |
        | add irrelevant conversational filler
        v
Transformed transcript

Expected relationship:

Important meeting facts should remain.
Important action items should remain.
The irrelevant filler should not create new facts/actions.
"""

from dataclasses import dataclass


@dataclass
class MetamorphicCase:
    case_id: str
    transformation_name: str
    original_input: str
    transformed_input: str


def append_irrelevant_filler(transcript: str) -> str:
    """
    Add conversational content that should not affect the
    important summary or action items.
    """

    filler = """

Additional unrelated conversational filler:

Before everyone leaves, there was also a brief discussion about
whether the office coffee machine should be replaced next quarter.
No decision was made, no owner was assigned, and this topic is
unrelated to the meeting objectives above.
"""

    return transcript.rstrip() + filler


def build_metamorphic_case(
    case_id: str,
    transcript: str,
) -> MetamorphicCase:

    transformed = append_irrelevant_filler(transcript)

    return MetamorphicCase(
        case_id=case_id,
        transformation_name="append_irrelevant_filler",
        original_input=transcript,
        transformed_input=transformed,
    )