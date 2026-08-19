import json
import re
from pathlib import Path

PAST_POSTS_FILE = Path("past_posts_log.json")

# words that appear capitalised mid-sentence but aren't factual claims
_COMMON_TERMS = {
    "ai", "ml", "llm", "api", "llms", "apis", "sdk", "ux", "ui", "ci", "cd",
    "gpu", "cpu", "saas", "erp", "crm", "eda", "pcb", "kpi", "kpis", "iot",
    "nlp", "aws", "gcp", "rag", "sql", "etl", "devops",
    "i", "we", "they", "he", "she", "the", "this", "that", "with", "and",
    "for", "not", "but", "from", "into", "open", "source", "via", "new",
    "now", "its", "also", "their", "more", "it", "is", "in", "on", "of",
    "to", "a", "an", "at", "by", "or", "be", "mit", "q1", "q2", "q3", "q4",
}


def grade_duplicate_url(story_url: str) -> dict:
    if not story_url:
        return {"passed": False, "reason": "story_url is empty"}
    if not PAST_POSTS_FILE.exists():
        return {"passed": True, "reason": ""}
    used = {e.get("source_url", "") for e in json.loads(PAST_POSTS_FILE.read_text())}
    if story_url in used:
        return {"passed": False, "reason": f"already posted: {story_url}"}
    return {"passed": True, "reason": ""}


def _proper_nouns(text: str) -> set[str]:
    # capitalised words mid-sentence only; skip numbers and common terms
    tokens = set()
    for sent in re.split(r'(?<=[.!?\n])\s+', text):
        for i, w in enumerate(sent.split()):
            if i == 0:
                continue
            clean = re.sub(r"[^a-zA-Z\-']", "", w).strip("-'")
            if len(clean) >= 3 and clean[0].isupper() and clean.lower() not in _COMMON_TERMS:
                tokens.add(clean.lower())
    return tokens


def grade_name_in_facts(post: str, source_facts: list[str]) -> dict:
    # numbers excluded — LLM critic handles stat fabrication
    facts = " ".join(source_facts).lower()
    invented = [
        t for t in sorted(_proper_nouns(post))
        if t not in facts and t.rstrip("'s").rstrip("s'") not in facts
    ]
    if invented:
        return {"passed": False, "reason": f"not in source_facts: {', '.join(invented[:10])}"}
    return {"passed": True, "reason": ""}


def run_all(labels_path: str = "eval_labels.json") -> None:
    rows = json.loads(Path(labels_path).read_text())
    dup_fails = name_fails = total = 0

    for row in rows:
        if row.get("fail_list_passed") is None:
            continue
        total += 1
        rid = row["id"]

        dup = grade_duplicate_url(row.get("story_url", ""))
        if not dup["passed"]:
            dup_fails += 1
            print(f"[dup-url  FAIL] id {rid}: {dup['reason']}")

        name = grade_name_in_facts(row.get("post", ""), row.get("source_facts", []))
        if not name["passed"]:
            name_fails += 1
            print(f"[name-inv FAIL] id {rid}: {name['reason']}")

    print(f"\ngraders summary ({total} rows)")
    print(f"  duplicate_url : {dup_fails} fails")
    print(f"  name_in_facts : {name_fails} fails")
    print(f"  pass rate: {(total * 2 - dup_fails - name_fails) / (total * 2):.0%}")


if __name__ == "__main__":
    run_all()
