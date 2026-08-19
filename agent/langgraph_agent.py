import difflib
import json
import os
from datetime import datetime
from pathlib import Path

import yaml
from tavily import TavilyClient
from dotenv import load_dotenv
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages, AnyMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from prompts import (
    get_user_posts_messages,
    get_outline_messages,
    get_critic_messages,
    get_rank_messages,
    get_research_messages,
)


_ = load_dotenv()

_DIR = Path(__file__).parent   # agent/
_ROOT = _DIR.parent             # repo root

"""
order flow:
fetch my_profile.txt + my obsidian notes/ideas -> fetch other people posts (two guys) -> planner (pick a topic, and angle from the data we have)
-> write_post (with the context of the data we gathered) -> critic
if pass -> print
if fail -> write it again (back to write_post)  
"""


llm = init_chat_model("gpt-4o")
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

PAST_POSTS_FILE = str(_ROOT / "past_posts_log.json")
EVAL_RUNS_FILE = str(_ROOT / "eval_runs.json")


class Post(BaseModel):
    post: str = Field(description="The post that is made by the agent")

class Planner(BaseModel):
    topic: str = Field(description="One line, the general subject of the post")
    angle: str = Field(description="one sentence, the SPECIFIC narrow take this post will center on")
    supporting_facts: list[str] = Field(description="2-4 literal concrete facts from the profile — numbers, tool names, decisions, outcomes — used as the raw material for the post")
    chosen_pattern: str = Field(description="the exact name of the structural pattern from the pattern library that best fits this topic/facts")

class Critic(BaseModel):
    passed: bool = Field(description="True if the post has no AI-slop patterns, False if it fails any rule")
    feedback: str = Field(description="If failed: specific, actionable feedback quoting the exact bad phrase and the rule it breaks. Empty string if passed.")

class Ranked(BaseModel):
    index: int = Field(description="index of the chosen candidate story, as numbered in the list")
    reason: str = Field(description="one sentence on why this story is worth posting about")

class Research(BaseModel):
    summary: str = Field(description="2-3 sentences on what the source actually says")
    facts: list[str] = Field(description="3-6 concrete facts stated in the source material — numbers, versions, names, specific decisions")


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    my_profile: str
    niche: str
    patterns: list[dict]
    candidates: list[dict]      # discovered stories, before ranking
    chosen_story: dict | None   # the one that won the ranking
    research: Research | None   # verified facts extracted from the chosen story
    past_posts_log: str
    outline: Planner | None
    post_made: str | None
    feedback: str | None        # critic's note — overwritten every critic run
    user_feedback: str | None   # human's note — persists across critic retries
    attempt_count: int
    approval_status: str | None


def fetch_my_profile(state: State):
    _ = state  # required by LangGraph node signature; unused for now
    print("Fetching your posts...")
    with open(_ROOT / "my_profile.txt", "r") as f:
        content = f.read()

    return {"my_profile": content}


def fetch_patterns(state: State):
    _ = state
    print("loading pattern library")

    with open(_DIR / "patterns.json", "r") as f:
        data = json.load(f)

    return {"patterns": data["patterns"]}


def find_pattern(patterns: list[dict], name: str) -> dict:
    """Look up a pattern by name, tolerating near-misses from the LLM.

    The planner sometimes returns a slightly wrong name ('notice-' vs 'noticed-'),
    which used to silently fall back to patterns[0] — a totally unrelated shape.
    """
    exact = next((p for p in patterns if p["name"] == name), None)
    if exact:
        return exact

    close = difflib.get_close_matches(name, [p["name"] for p in patterns], n=1, cutoff=0.6)
    if close:
        print(f"  [pattern] '{name}' not exact, matched -> '{close[0]}'")
        return next(p for p in patterns if p["name"] == close[0])

    print(f"  [warning] pattern '{name}' unmatched, using first pattern")
    return patterns[0]


def discover(state: State):
    """Search the web for candidate stories inside the niche. No LLM — just retrieval."""
    _ = state
    print("discovering stories...")

    with open(_DIR / "niche.yml", "r") as f:
        niche_raw = f.read()
    niche_cfg = yaml.safe_load(niche_raw)

    past_posts_log = ""
    used_urls = set()
    if os.path.exists(PAST_POSTS_FILE):
        with open(PAST_POSTS_FILE, "r") as f:
            entries = json.load(f)
        past_posts_log = "\n".join(
            f"- {e['date']}: {e['topic']} ({e.get('source_url', 'no source')})" for e in entries
        )
        used_urls = {e.get("source_url", "") for e in entries if e.get("source_url")}

    candidates = []
    skipped = 0
    for query in niche_cfg.get("discovery_queries", []):
        try:
            results = tavily.search(query=query, max_results=3, topic="news", days=30)
        except Exception as e:
            print(f"  [search failed] {query}: {e}")
            continue
        for r in results.get("results", []):
            url = r.get("url", "")
            # already posted about this one — drop it before ranking ever sees it.
            # don't rely on the LLM to respect the log when code can enforce it.
            if url in used_urls:
                skipped += 1
                continue
            candidates.append({
                "title": r.get("title", ""),
                "url": url,
                "content": (r.get("content", "") or "")[:600],
            })

    print(f"  found {len(candidates)} candidates ({skipped} skipped as already posted)")
    return {"candidates": candidates, "niche": niche_raw, "past_posts_log": past_posts_log}


def rank(state: State):
    """Pick the single best candidate to post about."""
    print("ranking stories...")

    candidates = state.get("candidates", [])
    if not candidates:
        return {"chosen_story": None}

    listing = "\n\n".join(
        f"[{i}] {c['title']}\n{c['content'][:300]}" for i, c in enumerate(candidates)
    )

    structured_llm = llm.with_structured_output(Ranked)
    result = structured_llm.invoke(
        get_rank_messages(state.get("niche", ""), listing, state.get("past_posts_log", ""))
    )
    assert isinstance(result, Ranked)

    idx = result.index if 0 <= result.index < len(candidates) else 0
    chosen = candidates[idx]
    print(f"  picked: {chosen['title'][:70]}")
    print(f"  why: {result.reason}")

    return {"chosen_story": chosen}


def research(state: State):
    """Go deep on the chosen story and extract only facts present in the source."""
    print("researching...")

    story = state.get("chosen_story")
    if not story:
        return {"research": None}

    # pull fuller context than the discovery snippet
    context = story.get("content", "")
    try:
        deep = tavily.search(query=story["title"], max_results=3, include_raw_content=False)
        context += "\n\n" + "\n\n".join(
            (r.get("content", "") or "")[:1000] for r in deep.get("results", [])
        )
    except Exception as e:
        print(f"  [deep search failed] {e}")

    structured_llm = llm.with_structured_output(Research)
    result = structured_llm.invoke(
        get_research_messages(story["title"], story["url"], context)
    )
    assert isinstance(result, Research)

    print(f"  extracted {len(result.facts)} facts")
    return {"research": result}


def planner(state: State):

   print("Planning the outline for the post...")

   # my_profile disabled — the researched story is the grounding material now.
   # re-enable by restoring this line and the fetch_my_profile node/edges below.
   # my_profile = state.get("my_profile", "")
   patterns = state.get("patterns", [])
   patterns_library = json.dumps(patterns, indent=2)

   # the researched story is what planner plans FROM — it does not invent material
   research = state.get("research", None)
   story = state.get("chosen_story", None)
   if research and story:
       my_profile = (
           f"Story: {story['title']}\n"
           f"Source: {story['url']}\n\n"
           f"What it says: {research.summary}\n\n"
           "Verified facts from the source (use ONLY these, do not add any):\n"
           + "\n".join(f"- {f}" for f in research.facts)
       )
   else:
       my_profile = ""

   # on a redo, planner sees what it planned last time + why the human rejected it,
   # so it can decide whether the SUBJECT needs to change or only the wording.
   user_feedback = state.get("user_feedback", "") or ""
   previous = state.get("outline", None)
   previous_outline = ""
   if previous and user_feedback:
       previous_outline = (
           f"topic: {previous.topic}\n"
           f"angle: {previous.angle}\n"
           f"facts: {previous.supporting_facts}\n"
           f"pattern: {previous.chosen_pattern}"
       )

   messages = get_outline_messages(
       my_profile,
       patterns_library,
       state.get("past_posts_log", ""),
       previous_outline,
       user_feedback,
   )

   structured_llm = llm.with_structured_output(Planner)
   planner = structured_llm.invoke(messages)

   print(f"  [topic] {planner.topic}")

   return {"outline": planner}

def log_eval_run(entry: dict) -> None:
    """Append one first-draft critic sample. File is a JSON list so you can label later."""
    entries = []
    if os.path.exists(EVAL_RUNS_FILE):
        with open(EVAL_RUNS_FILE) as f:
            entries = json.load(f)
    entries.append(entry)
    with open(EVAL_RUNS_FILE, "w") as f:
        json.dump(entries, f, indent=2)


def critic(state: State):
    print("critically analyzing the post")

    post = state.get("post_made", "")
    attempt_count = state.get("attempt_count", 0)
    outline = state.get("outline", None)
    patterns = state.get("patterns", [])

    pattern = find_pattern(patterns, outline.chosen_pattern) if outline else None
    if pattern is None:
        pattern = {"description": "", "example_post": ""}

    # give the critic the researched facts so it can verify every specific claim
    research = state.get("research", None)
    source_facts = "\n".join(f"- {f}" for f in research.facts) if research else ""

    critic_messages = get_critic_messages(
        post, pattern["description"], pattern["example_post"], source_facts
    )

    structured_llm = llm.with_structured_output(Critic)
    result = structured_llm.invoke(critic_messages)
    assert isinstance(result, Critic)

    # first uncoached draft only — log before pass/fail so both outcomes land in the set
    if attempt_count == 0:
        story = state.get("chosen_story") or {}
        log_eval_run({
            "logged_at": datetime.now().isoformat(timespec="seconds"),
            "story_title": story.get("title", ""),
            "story_url": story.get("url", ""),
            "outline": {
                "topic": outline.topic if outline else "",
                "angle": outline.angle if outline else "",
                "supporting_facts": outline.supporting_facts if outline else [],
                "chosen_pattern": outline.chosen_pattern if outline else "",
            },
            "post": post,
            "source_facts": research.facts if research else [],
            "chosen_pattern": outline.chosen_pattern if outline else "",
            "critic_messages": critic_messages,
            "critic_passed": result.passed,
            "critic_feedback": result.feedback,
            "human_passed": None,
        })
        print(f"  [eval] logged first draft -> {EVAL_RUNS_FILE}")

    if result.passed:
        print("  [critic] passed")
        return {"feedback": None}

    print(f"  [critic] failed: {result.feedback}")
    return {"feedback": result.feedback, "attempt_count": attempt_count + 1}


def route_after_critic(state: State):
    feedback = state.get("feedback")
    attempt_count = state.get("attempt_count", 0)

    if feedback and attempt_count < 3:
        return "make_a_post"
    return "human_approval"

def update_post(state: State):
    """Takes the human's raw feedback and frames it as an instruction for the rewrite."""
    print("processing your feedback...")

    raw = state.get("user_feedback", "")
    framed = f"The user has provided the following feedback, make the post better: {raw}"

    message = {"role": "user", "content": framed}

    # write to user_feedback (persists) — NOT feedback, which the critic overwrites.
    # also clear the stale critic note so the rewrite starts clean.
    return {"messages": [message], "user_feedback": framed, "feedback": None}


def human_approval(state: State):
    """Pauses the graph and waits for a human decision.

    NOTE: on resume this whole function re-runs from the top — LangGraph replays
    the node. So no side effects before interrupt().
    """
    post = state.get("post_made", "")

    # graph stops HERE. this value is handed back to whoever called invoke().
    decision = interrupt({"post": post})

    # only reached after resume, with decision = whatever was passed to Command(resume=...)
    if decision == "approve":
        return {"approval_status": "approved"}

    # anything else is treated as rejection feedback for a rewrite
    return {"approval_status": "rejected", "user_feedback": decision, "attempt_count": 0}


def route_after_approval(state: State):

    if state.get("approval_status") == "approved":
        return "save_to_memory"
    # rejected -> run the feedback through update_post before rewriting
    return "update_post"


def save_to_memory(state: State):
    """Log the approved post so future runs don't repeat the topic."""
    print("saving to memory...")

    outline = state.get("outline", None)
    story = state.get("chosen_story", None)

    entries = []
    if os.path.exists(PAST_POSTS_FILE):
        with open(PAST_POSTS_FILE, "r") as f:
            entries = json.load(f)

    entries.append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "topic": outline.topic if outline else "unknown",
        "angle": outline.angle if outline else "",
        "source_url": story["url"] if story else "",
        "post": state.get("post_made", ""),
    })

    with open(PAST_POSTS_FILE, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"  logged. {len(entries)} posts in history")
    return {}


def make_a_post(state: State):
    outline = state.get("outline", None)
    patterns = state.get("patterns", [])
    feedback = state.get("feedback", "") or ""
    user_feedback = state.get("user_feedback", "") or ""
    if not outline:
        return {"post_made": "no outline found, can't generate a post"}

    pattern = find_pattern(patterns, outline.chosen_pattern)

    print(f"  [pattern] using: {pattern['name']}")

    structured_llm = llm.with_structured_output(Post)
    post = structured_llm.invoke(
        get_user_posts_messages(
            outline.topic,
            outline.angle,
            outline.supporting_facts,
            pattern["description"],
            pattern["example_post"],
            feedback,
            user_feedback,
        )
    )
    assert isinstance(post, Post)
    return {"post_made": post.post}


graph_builder = StateGraph(State)

# graph_builder.add_node("fetch_my_profile", fetch_my_profile)   # profile disabled
graph_builder.add_node("fetch_patterns", fetch_patterns)
graph_builder.add_node("discover", discover)
graph_builder.add_node("rank", rank)
graph_builder.add_node("research", research)
graph_builder.add_node("save_to_memory", save_to_memory)
graph_builder.add_node("planner", planner)
graph_builder.add_node("make_a_post", make_a_post)
graph_builder.add_node("critic", critic)
graph_builder.add_node("human_approval", human_approval)
graph_builder.add_node("update_post", update_post)

# graph_builder.add_edge(START, "fetch_my_profile")                      # profile disabled
# graph_builder.add_edge(start_key="fetch_my_profile", end_key="planner")  # profile disabled

# linear. planner must NOT start before research finishes — with two incoming edges
# LangGraph fired planner as soon as fetch_patterns landed, skipping the research.
graph_builder.add_edge(START, "fetch_patterns")
graph_builder.add_edge("fetch_patterns", "discover")
graph_builder.add_edge("discover", "rank")
graph_builder.add_edge("rank", "research")
graph_builder.add_edge("research", "planner")
graph_builder.add_edge("planner", "make_a_post")
graph_builder.add_edge("make_a_post", "critic")
graph_builder.add_conditional_edges("critic", route_after_critic)
graph_builder.add_conditional_edges("human_approval", route_after_approval)
graph_builder.add_edge("update_post", "planner")
graph_builder.add_edge("save_to_memory", END)

# the checkpointer saves a snapshot of state after EVERY node runs.
# MemorySaver keeps them in a dict in this process — dies when the process exits.
# swap for SqliteSaver later to survive restarts.
checkpointer = MemorySaver()

graph = graph_builder.compile(checkpointer=checkpointer)


def run_agent(thread_id: str | None = None):
    print("Starting agent...")

    # thread_id identifies THIS run. same id later = resume that run's saved state
    # instead of starting over. new id = fresh run.
    thread_id = thread_id or f"post-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    config = {"configurable": {"thread_id": thread_id}}
    print(f"thread_id: {thread_id}")

    state: State = {
        "messages": [],
        "my_profile": "",
        "niche": "",
        "patterns": [],
        "candidates": [],
        "chosen_story": None,
        "research": None,
        "past_posts_log": "",
        "outline": None,
        "post_made": None,
        "feedback": None,
        "user_feedback": None,
        "attempt_count": 0,
        "approval_status": None,
    }

    result = graph.invoke(state, config)

    while "__interrupt__" in result:
        #this is the post that was made and needs approval
        paused_post = result["__interrupt__"][0].value["post"]

        print("\n" + "=" * 60)
        print(paused_post)
        print("=" * 60)
        answer = input("\napprove? (y = post it or suggest changes): ").strip()

        resume_value = "approve" if answer.lower() in ("y", "yes") else answer
        result = graph.invoke(Command(resume=resume_value), config)

    print(f"\n[{result.get('approval_status')}] final post:\n")
    print(result["post_made"])

    history = list(graph.get_state_history(config))
    print(f"\n[checkpointer] saved {len(history)} snapshots for this run")
    for snapshot in reversed(history):
        node = snapshot.metadata.get("step", "?") if snapshot.metadata else "?"
        next_up = snapshot.next or ("END",)
        print(f"  step {node}: next -> {next_up}")


if __name__ == "__main__":
    run_agent()