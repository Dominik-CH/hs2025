import os
from peewee import SqliteDatabase

path_to_db_file = os.path.join(os.path.dirname(__file__),'character_llm.db')
clip_db = SqliteDatabase(path_to_db_file)
