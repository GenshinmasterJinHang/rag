"""Offline wiring tests. Real-model checks are separate in verify_rag.py."""
import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage

import agent
import rag
import rag_tool


class FakeToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class WorkflowTests(unittest.TestCase):
    def test_agent_retrieves_even_when_model_never_requests_a_tool(self):
        evidence = "SOURCE: university_it_policies.txt | POLICY: doc-0001\nA supported fact."
        fake = FakeToolModel(responses=[AIMessage(content="A test answer.")])
        with patch("agent.make_model", return_value=fake), patch.object(
            agent.search_university_knowledge.__class__, "invoke", return_value=evidence
        ) as search:
            result = agent.answer_question("A policy question?")
        search.assert_called_once_with({"question": "A policy question?"})
        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual([m.content for m in tool_messages], [evidence])
        self.assertEqual(result["answer"], "A test answer.")

    def test_search_tool_preserves_policy_provenance(self):
        document = Document(
            page_content="Unreleased print jobs are retained for up to 24 hours.",
            metadata={"source": "university_it_policies.txt", "policy_id": "doc-0001", "topic": "printing"},
        )
        with patch("rag_tool.retrieve_documents", return_value=[document]):
            context = rag_tool.search_university_knowledge.invoke({"question": "Print retention?"})
        self.assertIn("24 hours", context)
        self.assertIn("university_it_policies.txt", context)
        self.assertIn("doc-0001", context)

    def test_empty_question_is_rejected_before_model_or_index_access(self):
        with patch("rag.get_vector_store") as store, patch("agent.make_model") as model:
            with self.assertRaises(ValueError):
                rag.retrieve_documents("   ")
            with self.assertRaises(ValueError):
                agent.answer_question("   ")
        store.assert_not_called()
        model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
