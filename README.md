# x-posting-agent

A LangGraph agent that finds engineering news, drafts an X post, critiques it against a no-slop rubric, and waits for human approval before logging it.

## How it works

```
fetch_patterns → discover → rank → research → planner → make_a_post → critic
                                                                          ↓
                                                              human_approval
                                                                  ↓       ↓
                                                           approved    rejected
                                                              ↓            ↓
                                                        save_to_memory   planner (rewrite)
```

1. **discover** — searches the web for fresh stories matching the niche (`agent/niche.yml`)
2. **rank** — picks the single best candidate via LLM
3. **research** — extracts 3-6 concrete facts from the source; the post can only use these
4. **planner** — picks a structural pattern (`agent/patterns.json`) and drafts an outline
5. **make_a_post** — writes the post following the chosen pattern
6. **critic** — rejects posts that hit any rule in the FAIL list (LinkedIn tone, fabrication, vague filler, closing moral, etc.)
7. **human_approval** — prints the post, waits for `y` or inline feedback; on rejection loops back to planner

## Structure

```
agent/          core pipeline — LangGraph graph, prompts, niche config, pattern library
evals/          evaluation infrastructure — critic replay, code graders, batch runner
scripts/        one-off utilities — extract patterns from reference posts
```

## Eval pipeline

The critic is measured against a human-labeled golden set.

```bash
make eval      # replay critic + run code graders, exits 1 if agreement < 70%
make graders   # code graders only (duplicate URL, name fabrication)
```

**Collecting samples**

```bash
PYTHONPATH=agent:. .venv/bin/python evals/run_eval_batch.py
```

Runs the agent N times headlessly, logs first drafts to `eval_runs.json`, and appends unlabeled rows to `eval_labels.json`.

**Labeling**

Open `eval_labels.json` and fill `fail_list_passed` (true/false) and `fail_list_reason` for each row, strictly against the critic's FAIL list — not whether you'd ship the post.

**Replaying the critic**

```bash
PYTHONPATH=agent:. .venv/bin/python evals/replay_critic.py
```

Re-runs the current critic prompt on every labeled draft and prints agreement per row.

## Running the agent

```bash
cp .env.example .env       # add OPENAI_API_KEY and TAVILY_API_KEY
python -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHONPATH=agent:. .venv/bin/python agent/langgraph_agent.py
```

## Runtime files (gitignored)

| File | Created by |
|---|---|
| `past_posts_log.json` | agent — approved posts, prevents repeats |
| `eval_runs.json` | agent — first drafts logged for eval |
| `eval_labels.json` | `run_eval_batch.py` — human labels go here |
| `my_profile.txt` | you — your background, fed to the planner |
