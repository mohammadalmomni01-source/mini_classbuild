from classbuild.course_config import validate_course_config
from classbuild.llm import GroqClientError
from classbuild.rag.embeddings import EmbeddingError, EmbeddingService
from classbuild.rag.pdf_processor import process_pdf
from classbuild.rag.retriever import RetrievalError, build_index
from classbuild.services.chat_state import append_exchange, get_history, thread_key
from classbuild.services.course_generator import CourseGenerator
from classbuild.services.lesson_chat import LessonChatContext, LessonChatService
from classbuild.services.lesson_generator import LessonGenerator
from classbuild.services.structured_output import CourseGenerationError

with open("sample.pdf", "rb") as file:
    document = process_pdf("sample.pdf", file.read())

embeddings = EmbeddingService()
index = build_index(document, embeddings)

config = validate_course_config("Introduction to Python", "Beginner", 3)
outline = CourseGenerator().generate_outline(config)
content = LessonGenerator().generate_lesson(outline, 2)
ctx = LessonChatContext.from_course(outline, 2, content)

service = LessonChatService(embeddings=embeddings)
chats = {}
key = thread_key("demo", content, document.fingerprint)  # the PDF thread of lesson 2

QUESTIONS = (
    "How do I add an item to a list in Python?",
    "What data types does Python have?",
    "What does the PDF say about Kubernetes?",
    "What is the difference between lists and tuples?",
)

for question in QUESTIONS:
    try:
        reply = service.answer(
            ctx, question, history=get_history(chats, key), document=document, index=index
        )
    except (CourseGenerationError, GroqClientError, EmbeddingError, RetrievalError) as exc:
        print(f"\n[FAILED] {question}\n  {exc}")
        break
    append_exchange(chats, key, reply.question, reply.answer)

    answer = reply.answer
    print(f"\nQ: {question}")
    print(f"Status: {answer.status}")
    print(f"A: {answer.content[:400]}")
    if answer.status == "answered":
        print("Cited pages:", answer.cited_pages)
        for source in answer.sources:
            print(f"  source {source.chunk_id}, page {source.page_number}, "
                  f"similarity {source.similarity:.3f}")
    else:
        scores = ", ".join(f"{passage.score:.2f}" for passage in reply.passages)
        print(f"  (passage scores: {scores})")

print("\nMessages stored:", len(get_history(chats, key)))