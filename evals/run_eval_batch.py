"""Run the posting graph N times to collect first drafts without human approval."""
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

from langgraph_agent import PAST_POSTS_FILE, EVAL_RUNS_FILE, graph

TARGET = 18
MAX_ATTEMPTS = 25
BACKUP = PAST_POSTS_FILE + ".bak"

EMPTY_STATE = {
    "messages": [], "my_profile": "", "niche": "", "patterns": [],
    "candidates": [], "chosen_story": None, "research": None,
    "past_posts_log": "", "outline": None, "post_made": None,
    "feedback": None, "user_feedback": None, "attempt_count": 0,
    "approval_status": None,
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def skip_urls() -> set[str]:
    urls = {e.get("source_url", "") for e in load_json(PAST_POSTS_FILE) if e.get("source_url")}
    urls |= {r.get("story_url", "") for r in load_json(EVAL_RUNS_FILE) if r.get("story_url")}
    return {u for u in urls if u}


def write_skip_log(urls: set[str]) -> None:
    # temporarily mark all known URLs as posted so discover skips them
    posts = load_json(BACKUP)
    seen = {e.get("source_url", "") for e in posts}
    for url in urls:
        if url and url not in seen:
            posts.append({"date": "eval-skip", "topic": "skip", "angle": "", "source_url": url, "post": ""})
            seen.add(url)
    write_json(PAST_POSTS_FILE, posts)


def pop_last_eval_row(url: str) -> None:
    entries = load_json(EVAL_RUNS_FILE)
    if entries and entries[-1].get("story_url") == url:
        entries.pop()
        write_json(EVAL_RUNS_FILE, entries)


def label_row(i: int, run: dict) -> dict:
    # strip critic verdict so labeler can't see the answer
    return {
        "id": i,
        "logged_at": run["logged_at"],
        "story_title": run["story_title"],
        "story_url": run["story_url"],
        "outline": run["outline"],
        "post": run["post"],
        "source_facts": run["source_facts"],
        "chosen_pattern": run["chosen_pattern"],
        "human_passed": None,
        "human_reason": "",
        "fail_list_passed": None,
        "fail_list_reason": "",
    }


def main() -> None:
    shutil.copy(PAST_POSTS_FILE, BACKUP)
    collected: list[str] = []
    blocked = skip_urls()
    write_skip_log(blocked)

    try:
        attempts = 0
        while len(collected) < TARGET and attempts < MAX_ATTEMPTS:
            attempts += 1
            before = len(load_json(EVAL_RUNS_FILE))
            thread_id = f"eval-batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{attempts}"
            config = {"configurable": {"thread_id": thread_id}}
            print(f"\n=== attempt {attempts} ({len(collected)}/{TARGET}) ===")
            try:
                graph.invoke(dict(EMPTY_STATE), config)
            except Exception as e:
                print(f"  failed: {e}")
                time.sleep(2)
                continue

            story = graph.get_state(config).values.get("chosen_story") or {}
            url = story.get("url", "")
            after = len(load_json(EVAL_RUNS_FILE))
            print(f"  {story.get('title', '')[:80]}")
            print(f"  url: {url}  logged: {after > before}")

            if after <= before:
                print("  skip — nothing logged")
                continue
            if not url or url in blocked:
                print("  skip — duplicate url")
                pop_last_eval_row(url)
                continue

            blocked.add(url)
            collected.append(url)
            write_skip_log(blocked)
            print(f"  kept ({len(collected)}/{TARGET})")
            time.sleep(1)
    finally:
        shutil.move(BACKUP, PAST_POSTS_FILE)

    runs = load_json(EVAL_RUNS_FILE)
    labels = load_json("eval_labels.json")
    known = {row["logged_at"] for row in labels}
    new_runs = [r for r in runs if r["logged_at"] not in known]
    start_id = max((row["id"] for row in labels), default=-1) + 1
    for i, run in enumerate(new_runs):
        labels.append(label_row(start_id + i, run))
    write_json("eval_labels.json", labels)
    print(f"\ndone: {len(collected)} new drafts appended to eval_labels.json")


if __name__ == "__main__":
    main()
