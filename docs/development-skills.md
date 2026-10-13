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
