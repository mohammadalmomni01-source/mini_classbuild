# Mini ClassBuild

Mini ClassBuild turns any topic into a structured course with interactive lessons, quizzes, RAG-grounded PDF references, conversational Q&A, and persistent local storage.

---

## 1. Overview
Mini ClassBuild is an educational assistant designed to build structured, interactive learning courses. The system guarantees trustworthy behavior through strict Pydantic validation, bounded retries, offline unit tests, and persistent local storage that survives browser refreshes.

## 2. Technology Stack
- **Web UI:** Streamlit
- **LLM Provider:** Groq Cloud API (`openai/gpt-oss-20b`)
- **Embeddings & Search:** `sentence-transformers/all-MiniLM-L6-v2`, `numpy`
- **Data Validation:** Pydantic v2
- **PDF Processing:** `pypdf`
- **Persistence:** Local JSON file store (Atomic file operations)

## 3. Requirements
- Python 3.11 or newer
- Operating System: Windows, macOS, or Linux
- A free Groq Cloud API key from [console.groq.com](https://console.groq.com/)

## 4. Environment
Ensure your Python installation is version 3.11+:
```powershell
python --version
```

## 5. Installation
Create and activate a virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts execution: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`)*

Install the dependencies:
```powershell
python -m pip install -r requirements.txt
```

## 6. Environment Configuration (.env)
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
Edit `.env` and provide your Groq API key:
```env
GROQ_API_KEY=gsk_your_actual_key_here
GROQ_MODEL=openai/gpt-oss-20b
GROQ_MAX_RETRIES=1
MINICLASSBUILD_SAVE_DIR=saved_courses
```

## 7. Launching the Application
Run the Streamlit app:
```powershell
python -m streamlit run app.py
```

## 8. Features & Workflow

### 8.1 Course Outline Generation
- Enter a topic, choose difficulty (`Beginner`, `Intermediate`, `Advanced`), and select lesson count (1–8).
- The outline is strictly validated before display.
- If a new generation fails, the previous valid course is retained.

### 8.2 Lesson Generation
- Generate lessons with clear explanations, code examples, summaries, and exercise questions with hints.
- Select lessons individually from the dropdown.

### 8.3 Multiple-Choice Quizzes
- Generate 5-question multiple-choice quizzes with 4 options per question.
- Instant client-side grading with explanations and retake capability.

### 8.4 PDF Learning Materials & Semantic Search (RAG)
- Upload reference PDFs (up to 10 MB).
- Chunks text by page and performs semantic retrieval using local vector embeddings.

### 8.5 Grounded Lessons with Checked Citations
- Ground lessons in retrieved PDF passages.
- Sections cite exact chunk IDs with verified page numbers. Uncited page mentions are strictly rejected.

### 8.6 Contextual Lesson Chatbot
- Ask follow-up questions per lesson.
- Normal mode draws context from lesson text and history.
- PDF mode answers solely from retrieved evidence, citing pages and refusing when evidence is insufficient.

### 8.7 Course Saving and Restoring
- Save and update courses locally in `saved_courses/`.
- Restoring replaces the screen without any Groq calls, resetting quiz answers and chats.
- Loaded PDF courses require re-uploading the original matching PDF to unlock PDF generation.

### 8.8 Markdown Export
- Download the complete course as a single clean Markdown file (`.md`), including outlines, lessons, quizzes, and a complete Answer Key.

## 9. Running Tests
All tests run offline using mocks and fakes:
```powershell
python -m unittest discover -s tests -t . -v
```

## 10. Running the Offline Evaluation Suite
Run all 25 baseline verification scenarios:
```powershell
python scripts/run_evaluation.py
```
*(Use `--no-json` to skip saving JSON reports to `evaluation_results/`)*

## 11. Final Verification Script
Verify the entire system with one command:
```powershell
python scripts/final_check.py --verbose
```

## 12. Technical Documentation
For deep technical explanations of prompts, retry budgets, embeddings, chunking algorithms, and persistence models, see:
- [docs/how_it_works.md](docs/how_it_works.md)
- [docs/validation_audit.md](docs/validation_audit.md)

## 13. Data Privacy & Saved Files
Saved course files (`saved_courses/course-<id>.json`) store structured outlines, generated lesson content, quizzes, and PDF metadata (filename and SHA-256 fingerprint). They **never** contain:
- API keys or secrets
- User chat messages or thread history
- Submitted quiz answers
- Raw PDF files or embedding vectors

## 14. Troubleshooting
- **API Key Rejected:** Ensure your key starts with `gsk_` and has no extra spaces in `.env`.
- **Streamlit Session State Warnings:** These only appear when running bare test scripts and can be ignored.
- **Port In Use:** Run `streamlit run app.py --server.port 8502` to use an alternate port.

## 15. Known Limitations
- Scanned (image-only) PDFs cannot be read (OCR is not included).
- Chatbot history is bounded to recent turns to stay within prompt budgets.
