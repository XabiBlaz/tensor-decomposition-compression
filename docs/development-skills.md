# Development skills and verification

Installed from original upstream sources on 2026-10-13 into the user's Codex
skills directory, outside this repository. Pins identify the reviewed snapshot;
updates are explicit. Skills guide work; tests and measured artifacts establish
correctness. No skill is treated as a runtime security boundary.

| Purpose | Source | Reviewed commit |
| --- | --- | --- |
| Requirements interview | [Matt Pocock grill-me and grilling](https://github.com/mattpocock/skills/tree/49dd158d1076134a641b33efb035946536778336/skills/productivity) | `49dd158d1076134a641b33efb035946536778336` |
| UI design, usability and accessibility | [Impeccable Codex build](https://github.com/pbakaus/impeccable/tree/2f98774f86bc05563876643db37e6c37c322850f/.agents/skills/impeccable) | `2f98774f86bc05563876643db37e6c37c322850f` |
| Fresh verification evidence | [Superpowers verification-before-completion](https://github.com/obra/superpowers/tree/bb92a77741419a4ab5f06e711a283343f1ada0c3/skills/verification-before-completion) | `bb92a77741419a4ab5f06e711a283343f1ada0c3` |
| Debugging from reproduced evidence | [Superpowers systematic-debugging](https://github.com/obra/superpowers/tree/bb92a77741419a4ab5f06e711a283343f1ada0c3/skills/systematic-debugging) | `bb92a77741419a4ab5f06e711a283343f1ada0c3` |

No uniquely identifiable upstream `better-ui` skill was found. Impeccable was
chosen for its dedicated existing-interface usability, accessibility and design
guidance. Installation does not mean its UI launcher or design changes were run.
The existing Ponytail guidance continues to favor cohesive changes and reuse.

Before completion: inspect actual diffs/callers, run focused checks, run the
feasible offline CPU suite, inspect skips/failures and run nonwriting dry runs.
Report exact commands and results. GPU tests stay explicit external commands.

Requirements already settled by the user: open models on NVIDIA GPUs, LLM first,
representative capability metrics, best measured trade-off, then CNN, ASR and TTS.
The user's instruction to proceed authorizes this bounded first implementation;
it does not authorize downloads/training locally, paid compute or publication.

## Repository skills in a new Cloud task

The repository vendors Ponytail and Frontend Design under `.agents/skills/`.
The [upstream pins](../.agents/skills/UPSTREAM.md) link to both reviewed
revisions and identify the licenses shipped with them.

Start a **new** Codex Cloud task for this repository and select
`codex/experiment-planner` as its initial checkout. Ask it to report the
available repository skills and their paths before reading any skill files.
The discovery check passes when its startup catalog lists both `ponytail`
and `frontend-design` from the checkout. Then ask it to read both skills
and confirm the path it used. If either is absent, use the following check in
the repository root to distinguish a missing checkout file from a discovery
issue.

```bash
git branch --show-current
git ls-files .agents/skills
test -r .agents/skills/ponytail/SKILL.md &&
  test -r .agents/skills/frontend-design/SKILL.md
```
