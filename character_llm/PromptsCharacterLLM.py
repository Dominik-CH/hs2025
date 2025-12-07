import os
from os import path
from typing import Dict
from typing import List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from pydantic import ValidationError


def get_template(file) -> str:
    with open(path.join(os.path.dirname(__file__), "..", "synthetic_dialogue", "resources", "prompts", file), mode="r",
              encoding="UTF-8") as f:
        content = "".join(line for line in f.readlines())
    return content


class BaseModelPrompt(BaseModel):
    @classmethod
    def get_json_out_parser(cls) -> JsonOutputParser:
        return JsonOutputParser(pydantic_object=cls)


def validate_json_with_model(target_model: BaseModelPrompt, data: Dict):
    try:
        return target_model(**data)
    except ValidationError as e:
        raise ValidationError
    except Exception as ex:
        raise ex


class Message(BaseModelPrompt):
    role: str = Field(default="",
                      description="Either Harry or User, depending on who said or thought something."
                      )
    utterance: str = Field(
        description="Your response to the previous message with your character description in mind and the aim to push "
                    "the conversation forward. THe conversation is between Harry Potter and the user conversing with Harry about the supplied event."
                    "Write the conversation as if Harry is talking to the user about what happened in this particular situation and how he felt about it etc."
    )


class Messages(BaseModelPrompt):
    conversation: List[Message] = Field(description="A list of at least 30 messages sent by Harry Potter and the User, "
                                                    "with respect to all relevant plot points. ")


class Event(BaseModelPrompt):
    event_context: str = Field(default="",
                       description="Include a summary of the single event being described, specifying exactly what happens and who is involved. Describe how Harry Potter and the other character(s) feel about each other within this one event only. Are they on good terms in this moment? What might Harry think about the person in front of him right now? Describe the tone of the interaction between Harry and the other party as it occurs in this specific event."
                       )


class GetEventsPrompt(PromptTemplate):
    event_summary: str = Field(default="", )

    class OutputModel(Event):
        events: List[Event] = Field(default="",
                                    description="A List of all relevant events that happen in the supplied chapter."
                                                "Only return the json object DO NOT USE MARKDOWN!"
                                    )

    def __init__(self, chapter_text: str):
        super().__init__(
            template=get_template(path.join("extract_events")),
            input_variables=["chapter_text"],
            partial_variables={
                "format_instructions": self.OutputModel.get_json_out_parser().get_format_instructions()
            }
        )
        object.__setattr__(self, "payload",
                           {
                               "chapter_text": chapter_text
                           })


class GetDialogueForEvent(PromptTemplate):
    event_summary: str = Field(default="", )

    class OutputModel(Messages):
        pass

    def __init__(self, event_context: str):
        super().__init__(
            template=get_template(path.join("generate_dialogue_for_event")),
            input_variables=["event_context"],
            partial_variables={
                "format_instructions": self.OutputModel.get_json_out_parser().get_format_instructions()
            }
        )
        object.__setattr__(self, "payload",
                           {
                               "event_context": event_context
                           })
