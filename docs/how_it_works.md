# How It Works: Technical Architecture

This document describes the technical architecture and core mechanisms of **Mini ClassBuild**.

---

## 1. System Architecture

Mini ClassBuild follows a strict unidirectional dependency structure:
```
app.py (Streamlit UI)
  │
  ▼
services/ (CourseGenerator, LessonGenerator, GroundedLessonGenerator, QuizGenerator, LessonChatService)
  │
  ├─► rag/ (pdf_processor, chunker, embeddings, retriever)
  ├─► llm/ (GroqClient, error handling, secret redaction)
  ├─► persistence/ (CourseStore, SavedCourse, atomic writes)
  └─► export/ (course_to_markdown, pure Python)
        │
        ▼
      models/ (CourseOutline, LessonContent, Quiz, ChatMessage)
```
- **Streamlit** is used strictly at the top layer (`app.py`). Services, models, RAG, and persistence modules never import Streamlit.
- **Pydantic v2** models define strict schemas with `extra="forbid"`. All AI responses and saved files must pass schema validation.

---

## 2. Prompts, Retries and Structured Output

### Prompt Design
Each AI feature uses a carefully crafted system prompt and user prompt:
- Prompts instruct the model to respond with **one JSON object only**, matching a specific schema.
- Reference passages from PDFs are labeled as untrusted data (`<passage id="..." ...>`) to prevent prompt injection.

### Two-Layer Retry Budget
1. **HTTP/SDK Retries (`groq_client.py`):**
   - Groq SDK automatically retries transient network errors (timeouts, rate limits, 5xx) up to `GROQ_MAX_RETRIES` times (default 1, clamped to 0–3).
   - Permanent client errors (400, 401, 403, 404) are failed immediately without retries.
2. **Application JSON Retry (`structured_output.py`):**
   - If the model returns malformed JSON or invalid output, `generate_validated` provides detailed error feedback to the model and allows **one** correction retry.
   - This prevents compounding retries from exceeding reasonable quotas.

---

## 3. PDF Ingestion, Chunking & Semantic Search (RAG)

1. **Extraction & Cleaning (`pdf_processor.py`):**
   - `pypdf.PdfReader` extracts page-by-page text.
   - Whitespace and line breaks are normalized while preserving code indentation.
   - Files are validated (size < 10 MB, `%PDF` header check, non-empty, unencrypted).
2. **Page-Aware Chunking (`chunker.py`):**
   - Text is split into overlapping chunks (default 800 chars, 150 chars overlap).
   - Natural separators (`\n\n`, `\n`, sentences, punctuation) are preferred.
   - Chunks never span across page boundaries, enabling exact page-level attribution.
3. **Embeddings & Search (`retriever.py`):**
   - Chunks are embedded using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
   - Cosine similarity identifies top-$k$ matching passages for any query.
   - Passages below `0.30` similarity are discarded. If no evidence exceeds the threshold, `InsufficientEvidenceError` is raised honestly without LLM hallucination.

---

## 4. Grounded Lessons & Interactive Chat

- **Citation Enforcement:** Grounded lesson sections must specify allowed `source_ids`. Citations are strictly cross-verified against retrieved passages.
- **No Uncited Page Mentions:** Mentioning "page 3" or "p. 3" in lesson text requires that page to be among the cited sources.
- **Chat State:** Chat history is managed by `LessonChatService` and bounded to recent turns to preserve context without blowing the prompt limit.

---

## 5. Course Persistence & Export

- **File Format:** Courses are stored locally as JSON in `saved_courses/course-<id>.json`.
- **ID Security:** IDs are randomly generated 32-hex-character UUIDs. Path traversal attacks (e.g., `../../`) are strictly rejected.
- **Atomic Operations:** Files are written to temporary files and replaced with `os.replace` to prevent partial/corrupted files during power cuts or crashes.
- **Privacy by Design:** Saved files contain course outlines, lessons, quizzes, and PDF metadata (filename and SHA-256 fingerprint). They **never** store API keys, chat history, quiz attempts, PDF binaries, or embedding vectors.
- **Markdown Export:** A pure Python function formats the course, ungenerated lesson placeholders, quizzes, and an Answer Key into readable Markdown for easy sharing.
