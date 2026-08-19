"""
Reads structure_reference.txt and catalogs EVERY post individually — one entry per post,
each with a name describing its structure and the full post content verbatim.

Runs standalone, NOT part of the main graph — run manually whenever structure_reference.txt changes.

Output: patterns.json — the library the planner picks a specific post to mirror from.
"""

import json

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

load_dotenv()

llm = init_chat_model("gpt-4o")


class Pattern(BaseModel):
    name: str = Field(description="short kebab-case name describing THIS post's structure, e.g. 'noticed-problem-then-explain-cost'")
    description: str = Field(description="how THIS specific post is built, line by line: how it opens, how the middle is organized, and exactly how it ends. Describe the real ending precisely — if it ends on a concrete detail, say so; never describe an ending as a 'broader conclusion' or 'lesson' unless it literally is one.")
    example_post: str = Field(description="the full post, copied VERBATIM from the reference file — every line, exact wording, nothing trimmed or paraphrased")
    best_for: str = Field(description="what kind of material this shape fits — e.g. 'a bug you hit and how you fixed it' or 'comparing two tools you tested'")


class PatternLibrary(BaseModel):
    patterns: list[Pattern] = Field(description="one entry per post in the reference file — do not merge, skip, or summarize any post")


def extract_patterns():
    with open("structure_reference.txt", "r") as f:
        reference_posts = f.read()

    system = """You catalog real X (Twitter) posts written by engineers with strong followings. These are the voice being copied — direct, concrete, unpolished. The opposite of LinkedIn-style corporate writing.

Your job: process EVERY post in the file, one entry each. Do NOT group posts together, do NOT skip any, do NOT keep the count "reasonable" — if there are 13 posts, produce 13 entries.

For each post:
- name: a short kebab-case name for how that post is built (based on its shape, not its topic)
- description: how that exact post is constructed — its opening move, how the middle is organized, and precisely how it ends. Be literal about the ending. If the post ends on a concrete technical detail, say that. If it ends with a short punchy line, say that. Only call an ending a "lesson" or "conclusion" if it genuinely generalizes beyond the specific work — most of these posts do NOT do that, and mislabeling it causes bad imitations later.
- example_post: the post copied VERBATIM. Every line, exact wording, original line breaks. This is the thing that gets imitated, so it must be exact — do not clean it up, do not paraphrase, do not trim.
- best_for: what kind of material would fit this shape

Posts are separated by '---' and grouped under account headers like '## @username'. Ignore the headers themselves — they aren't posts."""

    structured_llm = llm.with_structured_output(PatternLibrary)
    result = structured_llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": f"Reference posts:\n\n{reference_posts}\n\nCatalog every post individually."},
    ])

    assert isinstance(result, PatternLibrary)

    with open("patterns.json", "w") as f:
        json.dump(result.model_dump(), f, indent=2)

    print(f"Cataloged {len(result.patterns)} posts:")
    for p in result.patterns:
        print(f"  - {p.name}: {p.best_for}")
    print("\nSaved to patterns.json")


if __name__ == "__main__":
    extract_patterns()
