import json
import os
from collections import defaultdict
from typing import List, DefaultDict, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from character_llm.PromptsCharacterLLM import validate_json_with_model, GetEventsPrompt, Message, Event, \
    GetDialogueForEvent
from character_llm.database.models import Chapters, ExperienceCompletion, Events


class DialogGeneration:
    def __init__(self, open_ai_token: str = ""):
        self.open_ai_token = open_ai_token
        self.open_ai_model = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=open_ai_token)
        self.book_by_chapter: DefaultDict[int, str] = self.parse_books()

    @staticmethod
    def parse_books():
        book_by_chapter: DefaultDict[int, str] = defaultdict(str)
        df = pd.read_csv('../resources/harry_potter_books.csv')
        df = df[df['book'].str.startswith('Book 1')]
        for index, row in df.iterrows():
            chapter = int(row['chapter'].split("chap-")[1])
            book_by_chapter[chapter] += " " + row['text']
        return book_by_chapter

    def llm_request(self, prompt):
        chain = prompt | self.open_ai_model
        result = chain.invoke(prompt.payload).content
        to_validate = json.loads(result)
        validated_output = validate_json_with_model(target_model=prompt.OutputModel,
                                                    data=to_validate)
        return validated_output

    def extract_important_event_and_information(self, chapter_text: str) -> List:
        result: GetEventsPrompt.OutputModel = self.llm_request(prompt=GetEventsPrompt(chapter_text=chapter_text))
        return result.events

    def parse_dialogue(self, dialogue: List[Message], context: str = ""):
        dialogue_text: str = "{}".format(context)
        for message in dialogue:
            dialogue_text += f"{Message.character_name}: {message.utterance}\n"

    def generate_dialogue(self, event: Event):
        prompt = GetDialogueForEvent(event_context=event.event_context)
        dialogue = self.llm_request(prompt=prompt)
        return dialogue

    def generate_dialogues(self, skip_chapters: Tuple[int] = ()):
        all_events: DefaultDict[int, List[Event]] = defaultdict()
        for chapter_num, chapter_text in self.book_by_chapter.items():
            if chapter_num in skip_chapters:
                continue
            chapter, created = Chapters.get_or_create(name=chapter_num, chapter_text=chapter_text)
            events: List[Event] = self.extract_important_event_and_information(chapter_text=chapter_text)
            all_events[chapter_num] = events
            for event_obj in events:
                event, created = Events.get_or_create(chapter=chapter, event_context=event_obj.event_context)
                conversation = self.generate_dialogue(event=event)
                ExperienceCompletion.get_or_create(conversation=conversation.model_dump_json(),
                                                   introduction=event.event_context,
                                                   event=event)


if __name__ == '__main__':
    load_dotenv(".env")
    token = os.getenv("OPENAI_API_KEY")
    idk = DialogGeneration(open_ai_token=token)
    idk.generate_dialogues(skip_chapters=(1,2,3,4,5,6,7,8,9,10,11,12,13))
