import hashlib
from typing import NamedTuple


class ChatThreadKey(NamedTuple):
    course_id: str
    lesson_number: int
    lesson_fingerprint: str
    document_fingerprint: str | None  # None = lesson mode

    @property
    def is_pdf(self):
        return self.document_fingerprint is not None


def lesson_fingerprint(content):
    """Changes whenever the lesson text changes (for example when it is regenerated)."""
    return hashlib.sha256(content.model_dump_json().encode("utf-8")).hexdigest()[:16]


def thread_key(course_id, content, document_fingerprint=None):
    return ChatThreadKey(
        course_id, content.lesson_number, lesson_fingerprint(content), document_fingerprint
    )


def get_history(chats, key):
    return list(chats.get(key, []))


def append_exchange(chats, key, question, answer):
    """Add the question and its answer together, so a failed request leaves no lone question."""
    chats[key] = get_history(chats, key) + [question, answer]


def clear_thread(chats, key):
    chats.pop(key, None)


def valid_thread_keys(course_id, lessons, document_fingerprint):
    keys = set()
    for content in lessons.values():
        keys.add(thread_key(course_id, content, None))
        if document_fingerprint is not None:
            keys.add(thread_key(course_id, content, document_fingerprint))
    return keys


def prune_chats(chats, valid_keys):
    for key in [key for key in chats if key not in valid_keys]:
        del chats[key]