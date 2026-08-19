import json
import sys

from graders import grade_duplicate_url, grade_name_in_facts
from langgraph_agent import Critic, find_pattern, llm
from prompts import get_critic_messages

# CI fails if human↔critic agreement drops below this
AGREEMENT_FLOOR = 0.70


def run_critic_replay(labels, runs, patterns) -> tuple[int, int]:
    structured = llm.with_structured_output(Critic)
    agree = total = 0

    for row, run in zip(labels, runs, strict=True):
        if row.get("fail_list_passed") is None:
            continue
        total += 1

        pattern = find_pattern(patterns, row["chosen_pattern"])
        source_facts = "\n".join(f"- {f}" for f in row["source_facts"])
        result = structured.invoke(
            get_critic_messages(row["post"], pattern["description"], pattern["example_post"], source_facts)
        )
        assert isinstance(result, Critic)
        match = result.passed == row["fail_list_passed"]
        agree += int(match)
        mark = "ok" if match else "MISS"
        print(f"  [{mark}] id {row['id']}: you={row['fail_list_passed']} critic={result.passed}")
        if not match and result.feedback:
            print(f"       {result.feedback[:140]}")

    return agree, total


def run_code_graders(labels) -> tuple[int, int]:
    fails = total = 0

    for row in labels:
        if row.get("fail_list_passed") is None:
            continue
        total += 1

        dup = grade_duplicate_url(row.get("story_url", ""))
        if not dup["passed"]:
            fails += 1
            print(f"  [dup-url  FAIL] id {row['id']}: {dup['reason']}")

        name = grade_name_in_facts(row.get("post", ""), row.get("source_facts", []))
        if not name["passed"]:
            fails += 1
            print(f"  [name-inv FAIL] id {row['id']}: {name['reason']}")

    return fails, total


def main() -> None:
    with open("patterns.json") as f:
        patterns = json.load(f)["patterns"]
    with open("eval_labels.json") as f:
        labels = json.load(f)
    with open("eval_runs.json") as f:
        runs = json.load(f)

    print("--- critic replay ---")
    agree, critic_total = run_critic_replay(labels, runs, patterns)
    rate = agree / critic_total if critic_total else 0

    print("\n--- code graders ---")
    grader_fails, grader_total = run_code_graders(labels)

    print(f"\nlabeled : {critic_total}")
    print(f"agreement : {agree}/{critic_total} ({rate:.0%})  floor={AGREEMENT_FLOOR:.0%}")
    print(f"grader fails : {grader_fails} / {grader_total * 2}")

    failed = rate < AGREEMENT_FLOOR or grader_fails > 0
    print("\nFAIL" if failed else "\nPASS")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
