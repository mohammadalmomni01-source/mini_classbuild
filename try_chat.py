from classbuild.course_config import validate_course_config
from classbuild.llm import GroqClientError
from classbuild.services.chat_state import append_exchange, get_history, thread_key
from classbuild.services.course_generator import CourseGenerator
from classbuild.services.lesson_chat import ChatInputError, LessonChatContext, LessonChatService
from classbuild.services.lesson_generator import LessonGenerator
from classbuild.services.structured_output import CourseGenerationError

config = validate_course_config("Introduction to Python", "Beginner", 3)
outline = CourseGenerator().generate_outline(config)
content = LessonGenerator().generate_lesson(outline, 2)
ctx = LessonChatContext.from_course(outline, 2, content)

service = LessonChatService()
chats = {}
key = thread_key("demo", content)

for question in ("Explain this lesson in simpler words.", "Can you give me another example?", "   "):
    try:
        reply = service.answer(ctx, question, history=get_history(chats, key))
    except ChatInputError as exc:
        print(f"\n[REJECTED] {exc}")
        continue
    except (CourseGenerationError, GroqClientError) as exc:
        print(f"\n[FAILED] {exc}")
        break
    append_exchange(chats, key, reply.question, reply.answer)
    print(f"\nQ: {question}\nA: {reply.answer.content[:300]}")

print("\nMessages stored:", len(get_history(chats, key)))