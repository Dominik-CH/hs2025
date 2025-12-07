from peewee import *

from character_llm.database.character_llm_db import clip_db


class BaseModel(Model):
    """Base model for all database tables."""
    class Meta:
        database = clip_db


class Chapters(BaseModel):
    id = AutoField()
    name = CharField(unique=True)
    chapter_text = TextField(unique=False)

    class Meta:
        table_name = 'chapters'


class Events(BaseModel):
    id = AutoField()
    chapter = ForeignKeyField(Chapters, backref='chapters')
    event_context = TextField(unique=False)

    class Meta:
        table_name = 'events'


class ExperienceCompletion(BaseModel):
    id = AutoField()
    introduction = TextField(null=True, default=None)
    conversation = TextField(null=True, default=None)
    event = ForeignKeyField(Events, backref='events', on_delete='CASCADE')  # Many-to-one


# Creating the tables
if __name__ == "__main__":
    clip_db.connect()
    clip_db.create_tables([Chapters, Events, ExperienceCompletion])
