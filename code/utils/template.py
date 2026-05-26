QUESTION_PLACEHOLDER = '[INSERT PROMPT HERE]'


def synthesis_message(template, task):
    if QUESTION_PLACEHOLDER not in template:
        return None

    return template.replace(QUESTION_PLACEHOLDER, task)
