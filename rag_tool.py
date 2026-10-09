"""The Lecture 11 search tool, backed by the single policy-file index."""
from langchain_core.tools import tool

from rag import format_documents, retrieve_documents


@tool
def search_university_knowledge(question: str) -> str:
    """Search university IT policies for facts relevant to a specific question.

    Related policies are not proof that a requested fact exists. If no passage
    explicitly supplies the requested information, the answer is unknown.
    """
    return format_documents(retrieve_documents(question))
