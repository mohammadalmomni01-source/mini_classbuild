import tempfile
import unittest
from unittest.mock import MagicMock, patch

import streamlit as st

import app
from classbuild.models.course import CourseOutline
from classbuild.models.lesson import LessonContent
from classbuild.models.quiz import Quiz
from classbuild.persistence import CourseStore, build_saved_course, new_course_id
from tests.fixtures import valid_lesson, valid_outline, valid_quiz


class AppTests(unittest.TestCase):
    def setUp(self):
        # Reset streamlit session state
        st.session_state.clear()
        app.init_state()

    def test_failed_request_preserves_previous_course(self):
        # Setup an existing valid outline
        outline = CourseOutline.model_validate(valid_outline(3))
        st.session_state[app.OUTLINE_KEY] = outline
        st.session_state[app.COURSE_ID_KEY] = "test-course-id"

        # Simulate form submission that raises an error in CourseGenerator
        with patch.object(app, "CourseGenerator") as MockGen:
            instance = MockGen.return_value
            instance.generate_outline.side_effect = RuntimeError("API failure")

            with patch("streamlit.form_submit_button", return_value=True), \
                 patch("streamlit.text_input", return_value="New Topic"), \
                 patch("streamlit.radio", return_value="Beginner"), \
                 patch("streamlit.slider", return_value=3), \
                 patch("streamlit.error") as mock_err, \
                 patch("streamlit.info") as mock_info:

                app.render_form()

                # Previous course outline MUST still be preserved!
                self.assertIsNotNone(st.session_state[app.OUTLINE_KEY])
                self.assertEqual(st.session_state[app.OUTLINE_KEY].title, outline.title)
                # An info/error notice must be given
                mock_info.assert_called()

    def test_stale_grounded_lessons_removed(self):
        # Fresh (non-loaded) course with grounded lesson
        st.session_state[app.SAVED_ID_KEY] = None
        st.session_state[app.DOC_KEY] = None  # No PDF currently loaded
        lesson = LessonContent.model_validate(valid_lesson(1, "Title 1", is_grounded=True))
        st.session_state[app.LESSONS_KEY] = {1: lesson}
        st.session_state[app.QUIZZES_KEY] = {}
        st.session_state[app.RESULTS_KEY] = {}
        st.session_state[app.ATTEMPTS_KEY] = {}

        with patch("streamlit.warning"):
            app.drop_stale_grounded_lessons()

        # The grounded lesson should be removed because doc is None
        self.assertNotIn(1, st.session_state[app.LESSONS_KEY])


class SaveLoadAppTests(unittest.TestCase):
    def setUp(self):
        st.session_state.clear()
        app.init_state()
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.store = CourseStore(directory=self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_saving_and_reopening_in_new_session_without_groq(self):
        outline = CourseOutline.model_validate(valid_outline(3))
        cid = new_course_id()
        course = build_saved_course(
            course_id=cid,
            topic="Intro to Python",
            outline=outline,
            lessons={},
            quizzes={},
        )
        self.store.save(course)

        # In a new session, loading should restore without any Groq call
        loaded = self.store.load(cid)
        app.restore_saved_course(loaded)
        self.assertEqual(st.session_state[app.OUTLINE_KEY].title, outline.title)
        self.assertEqual(st.session_state[app.SAVED_ID_KEY], cid)

    def test_loading_resets_quiz_answers_and_chats(self):
        st.session_state[app.RESULTS_KEY] = {1: "old_result"}
        st.session_state[app.CHATS_KEY] = {("thread", 1): ["msg"]}

        outline = CourseOutline.model_validate(valid_outline(3))
        course = build_saved_course(
            course_id=new_course_id(),
            topic="Python",
            outline=outline,
            lessons={},
            quizzes={},
        )
        app.restore_saved_course(course)

        self.assertEqual(st.session_state[app.RESULTS_KEY], {})
        self.assertEqual(st.session_state[app.CHATS_KEY], {})

    def test_identical_titles_saved_as_separate_files(self):
        outline = CourseOutline.model_validate(valid_outline(3))
        cid1 = new_course_id()
        cid2 = new_course_id()
        c1 = build_saved_course(course_id=cid1, topic="Python", outline=outline, lessons={}, quizzes={})
        c2 = build_saved_course(course_id=cid2, topic="Python", outline=outline, lessons={}, quizzes={})

        p1 = self.store.save(c1)
        p2 = self.store.save(c2)

        self.assertNotEqual(p1, p2)
        listing = self.store.list_courses()
        self.assertEqual(len(listing.courses), 2)

    def test_loaded_pdf_course_gates_unrelated_pdf(self):
        outline = CourseOutline.model_validate(valid_outline(3))
        lesson = LessonContent.model_validate(valid_lesson(1, outline.lessons[0].title, is_grounded=True))
        course = build_saved_course(
            course_id=new_course_id(),
            topic="Python",
            outline=outline,
            lessons={1: lesson},
            quizzes={},
        )
        app.restore_saved_course(course)

        # Required source document is set
        req = app.required_source_document()
        self.assertIsNotNone(req)
        self.assertEqual(req.fingerprint, "a" * 32)

        # Without uploaded PDF matching fingerprint, pdf_features_allowed is False
        st.session_state[app.DOC_KEY] = None
        self.assertFalse(app.pdf_features_allowed())

        # With different PDF
        fake_doc = MagicMock()
        fake_doc.fingerprint = "b" * 32
        st.session_state[app.DOC_KEY] = fake_doc
        self.assertFalse(app.pdf_features_allowed())

        # With matching PDF
        fake_doc.fingerprint = "a" * 32
        self.assertTrue(app.pdf_features_allowed())

    def test_damaged_saved_files_reported_in_panel(self):
        # Create damaged file
        bad_id = new_course_id()
        (self.store._dir / f"course-{bad_id}.json").write_text("{corrupt", encoding="utf-8")

        listing = self.store.list_courses()
        self.assertEqual(len(listing.errors), 1)

    def test_failed_save_leaves_state_unchanged(self):
        st.session_state[app.OUTLINE_KEY] = None
        # Attempting to save with no outline gives error
        with patch("streamlit.error") as mock_err:
            app.save_current_course(self.store)
            mock_err.assert_called()
        self.assertIsNone(st.session_state[app.SAVED_ID_KEY])

    def test_markdown_download_offered(self):
        outline = CourseOutline.model_validate(valid_outline(3))
        st.session_state[app.OUTLINE_KEY] = outline
        st.session_state[app.CONFIG_KEY] = {"topic": "Python"}
        with patch("streamlit.download_button") as mock_dl:
            app.render_outline()
            mock_dl.assert_called()


if __name__ == "__main__":
    unittest.main()
