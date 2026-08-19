"""Replay the current critic prompt against labeled first drafts and score agreement."""
import json

from langgraph_agent import Critic, find_pattern, llm
from prompts import get_critic_messages


def main() -> None:
    with open("agent/patterns.json") as f:
        patterns = json.load(f)["patterns"]
    with open("eval_labels.json") as f:
        labels = json.load(f)
    with open("eval_runs.json") as f:
        runs = json.load(f)

    structured = llm.with_structured_output(Critic)
    out = []
    agree = 0

    for row, run in zip(labels, runs, strict=True):
        pattern = find_pattern(patterns, row["chosen_pattern"])
        source_facts = "\n".join(f"- {f}" for f in row["source_facts"])
        result = structured.invoke(
            get_critic_messages(row["post"], pattern["description"], pattern["example_post"], source_facts)
        )
        assert isinstance(result, Critic)
        match = result.passed == row["fail_list_passed"]
        agree += int(match)
        out.append({
            "id": row["id"],
            "fail_list_passed": row["fail_list_passed"],
            "critic_v1_passed": run["critic_passed"],
            "critic_v2_passed": result.passed,
            "critic_v2_feedback": result.feedback,
            "agree_v2": match,
        })
        mark = "ok" if match else "MISS"
        print(f"[{mark}] id {row['id']}: you={row['fail_list_passed']} v1={run['critic_passed']} v2={result.passed}")
        if result.feedback:
            print(f"     {result.feedback[:160]}")

    print(f"\nagreement: {agree}/{len(out)} ({agree / len(out):.0%})")
    with open("eval_replay.json", "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
