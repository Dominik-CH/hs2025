from character_llm.database.models import ExperienceCompletion


def get_finished_experience_completions():
    return [clip for clip in ExperienceCompletion.select().where(~ExperienceCompletion.conversation.is_null())]


def get_all_dialogues():
    return list(
        ExperienceCompletion
        .select(ExperienceCompletion.conversation, ExperienceCompletion.introduction)
        .where(~ExperienceCompletion.conversation.is_null())
    )
