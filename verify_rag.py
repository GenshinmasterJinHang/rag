"""Integration checks using real retrieval and real Qwen3-0.6B responses."""
import json
from pathlib import Path
import re

import agent
import rag
from policy_loader import load_policy_documents


def verify():
    documents = load_policy_documents()
    assert len(documents) == 1196
    assert len({d.metadata["policy_id"] for d in documents}) == 1196
    assert {d.metadata["source"] for d in documents} == {"university_it_policies.txt"}
    corpus = "\n".join(d.page_content for d in documents)
    phone_pattern = r"(?<!\w)\+?\d[\d ()-]{5,}\d(?!\w)"
    assert re.search(phone_pattern, corpus) is None

    cases = [
        ("missing_phone", rag.DEFAULT_QUESTION),
        ("phone_paraphrase", "Please give me the IT help desk's telephone number."),
        ("printing", "How long are unreleased print jobs retained?"),
        ("guest_wifi", "Which Wi-Fi network should visitors use?"),
    ]
    records = []
    failures = []
    for mode, answer_fn in [("rag", rag.answer_question), ("agent", agent.answer_question)]:
        for case, question in cases:
            result = answer_fn(question)
            answer = result["answer"]
            lower = answer.lower()
            if case in {"missing_phone", "phone_paraphrase"}:
                unknown = any(s in lower for s in (
                    "don't know", "do not know", "not specified", "not provided",
                    "does not specify", "doesn't specify", "does not provide",
                    "not available", "not contain", "isn't specified",
                ))
                unrelated = any(s in lower for s in (
                    "you should contact", "please contact", "visit the", "check the website",
                    "password", "mailbox", "phishing", "email", "vpn", "print",
                    "wi-fi", "wifi", "minutes", "hours", "recommend", "try ",
                ))
                sentences = [s for s in re.split(r"[.!?]+\s*", answer) if s.strip()]
                passed = (unknown and not unrelated
                          and re.search(phone_pattern, answer) is None
                          and len(answer.split()) <= 35 and len(sentences) == 1)
            elif case == "printing":
                passed = "24" in answer and "hour" in lower
            else:
                passed = "campus-guest" in lower
            record = {"mode": mode, "case": case, "question": question, "answer": answer, "passed": passed}
            if "documents" in result:
                record["sources"] = [d.metadata for d in result["documents"]]
            else:
                record["trace"] = [m.model_dump(mode="json") for m in result["messages"]]
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if not passed:
                failures.append(f"{mode}/{case}")

    ids = rag.get_vector_store().get()["ids"]
    assert len(ids) == 1196 and len(set(ids)) == 1196
    output = Path(__file__).resolve().parent / "RAG_Test_Results.json"
    output.write_text(json.dumps({
        "model": rag.MODEL_NAME,
        "embedding_model": rag.EMBEDDING_MODEL,
        "policy_records": len(documents),
        "results": records,
        "passed": len(records) - len(failures),
        "total": len(records),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if failures:
        raise AssertionError("Failed real-model checks: " + ", ".join(failures))
    print(f"PASS: {len(records)}/{len(records)} real-model checks; {len(ids)} unique indexed policies.")


if __name__ == "__main__":
    verify()
