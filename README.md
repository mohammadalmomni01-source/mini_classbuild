# Mini ClassBuild

Mini ClassBuild turns a topic into a short, structured course. It can also use a PDF you upload as reference material, so lessons and chat answers can point back to real pages.

Built with Streamlit, the Groq API, and a small local embedding model.

## Features

- **Course outline**: enter a topic, a difficulty (Beginner, Intermediate, Advanced) and 2 to 5 lessons. The outline is checked before it is shown.
- **Lessons**: generate one lesson at a time, with an introduction, sections with examples, a summary and exercises with hints.
- **Quizzes**: generate a multiple-choice quiz for a lesson. It is graded on your computer, with no extra AI call, and you can retake it.
- **PDF upload**: upload a PDF (up to 10 MB). Text is extracted page by page and split into chunks of about 800 characters with 150 characters of overlap. Scanned or image-only PDFs are reported clearly.
- **Semantic search**: find the passages closest in meaning to a question, with page numbers and similarity scores.
- **PDF-grounded lessons**: tick "Use Uploaded PDF as Reference" and the lesson is written from retrieved passages. Every cited source id is checked against what was actually retrieved, and page numbers come from the PDF, never from the AI. If the PDF does not cover the lesson, the app says so.
- **Lesson chat**: ask follow-up questions about the current lesson.
  - *Normal mode* answers from the lesson and your recent messages.
  - *PDF mode* (when the PDF reference box is ticked) answers only from retrieved passages, with checked page citations. If the PDF lacks the evidence, you get an "insufficient evidence" reply instead of a made-up answer.
  - Each lesson has its own conversation. **Clear Chat** clears only the current lesson.

## How it fits together

```
app.py                      Streamlit screens
classbuild/
  course_config.py          validates topic, difficulty, lesson count
  llm/                      Groq client and clear error types
  models/                   Pydantic models: course, lesson, quiz, chat
  rag/                      PDF reading, chunking, embeddings, retrieval, saved state
  services/                 course, lesson, grounded lesson, quiz, chat logic
sample.pdf                  small Python PDF to try the PDF features
blank.pdf                   a PDF with no text, to try the "no extractable text" message
```

## Setup

You need Python 3.10 or newer and a free [Groq API key](https://console.groq.com/).

```bash
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# macOS / Linux:        source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your key:

```
GROQ_API_KEY=your_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

`.env` is ignored by git, so your key is never committed.

## Run

```bash
streamlit run app.py
```

The first PDF search or PDF-based lesson downloads the embedding model (`sentence-transformers/all-MiniLM-L6-v2`, about 90 MB), so it needs an internet connection once. Embeddings are computed on your computer, and the PDF itself is never sent to Groq as a file. Only the few retrieved passages are included in prompts.

## Try it

1. Generate a course, for example "Introduction to Python", Beginner, 3 lessons.
2. Choose a lesson and click **Generate Lesson Content**.
3. Under **Ask about this lesson**, ask "Explain this lesson in simpler words", then "Can you give me another example?".
4. Upload `sample.pdf` under **Learning materials**, then generate a **new course**. The PDF is attached to the course it was loaded with.
5. Tick **Use Uploaded PDF as Reference**, generate a lesson, and ask a question about the PDF. Then ask about something it does not cover, such as Kubernetes, and see the "insufficient evidence" reply.

## Design notes

- The AI never supplies sources. Source ids and page numbers are looked up from retrieval, and unknown ids are rejected.
- A passage must score at least 0.30 similarity to count as evidence. If nothing does, no Groq call is made for the answer.
- Generated JSON is validated with Pydantic, with one bounded retry when it is invalid or uncited.
- Chat history sent to Groq is limited to the last 6 exchanges. The current lesson is always included.
- A failed request never adds a half-finished exchange to the saved chat.
- Replacing the PDF removes PDF-based lessons and chats that belonged to the old file.

## Limits

- Similarity scores measure closeness of meaning. They do not prove an answer is correct.
- Quality depends on the Groq model you pick in `.env`.
- PDFs are read as text only. Scanned pages need OCR, which is not included.
