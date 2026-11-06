def stale_grounded_lessons(lessons, current_fingerprint):
    """Numbers of PDF-based lessons whose PDF is no longer the loaded one."""
    return [
        number
        for number, content in lessons.items()
        if content.is_grounded and content.grounding.document_fingerprint != current_fingerprint
    ]


def remove_stale_grounded_lessons(lessons, current_fingerprint, *dependent):
    """Remove them, plus their entries in dictionaries like quizzes and results."""
    stale = stale_grounded_lessons(lessons, current_fingerprint)
    for number in stale:
        lessons.pop(number, None)
        for mapping in dependent:
            mapping.pop(number, None)
    return stale