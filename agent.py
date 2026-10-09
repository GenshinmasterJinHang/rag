"""Tool-based university policy agent using the required Qwen3-0.6B model."""
import argparse

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from rag import DEFAULT_QUESTION, SYSTEM_PROMPT, make_model
from rag_tool import search_university_knowledge


def answer_question(question=DEFAULT_QUESTION):
    if not question.strip():
        raise ValueError("Question must not be empty.")
    agent = create_agent(
        model=make_model(),
        tools=[search_university_knowledge],
        system_prompt=SYSTEM_PROMPT + "\nUse search_university_knowledge for policy facts.",
    )
    # Always perform retrieval before answering. A small model must not be able
    # to skip the only evidence source by deciding not to call a tool.
    call_id = "initial_policy_search"
    evidence = search_university_knowledge.invoke({"question": question})
    messages = [
        HumanMessage(content=question),
        AIMessage(content="", tool_calls=[{
            "name": "search_university_knowledge",
            "args": {"question": question}, "id": call_id, "type": "tool_call",
        }]),
        ToolMessage(content=evidence, tool_call_id=call_id),
    ]
    result = agent.invoke({"messages": messages}, config={"recursion_limit": 8})
    final = result["messages"][-1]
    if final.response_metadata.get("done_reason") == "length":
        raise RuntimeError("The model response was truncated.")
    if not isinstance(final.content, str) or not final.content.strip():
        raise RuntimeError("The agent did not return a nonempty answer.")
    return {"question": question, "answer": final.content.strip(), "messages": result["messages"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    result = answer_question(args.question)
    print("FINAL ANSWER\n" + result["answer"])
    if args.trace:
        print("\nFULL AGENT TRACE")
        for message in result["messages"]:
            print("\n" + type(message).__name__)
            print(message)
