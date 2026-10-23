# Repository guidance

## Start with the current project context

Read [`agent_docs/README.md`](agent_docs/README.md), then follow its read order. For current work, read [`agent_docs/active/README.md`](agent_docs/active/README.md) and the linked active, state, blocker, and progress files.

`agent_docs/` is local-only and Git-ignored. Treat it as the current project context, but do not assume it is shared, backed up, or synced to the server. Do not duplicate its contents in another active-docs folder.

## Basic working notes

- Check the current code, saved artifacts, and Git status before making claims or changes. Active notes can lag behind the checkout.
- Preserve unrelated working-tree changes. Keep changes within the user's stated scope.
- Separate implementation checks, saved experiment results, interpretation, and proposed work. Do not present a prior receipt as a fresh run.
- Do not launch experiments or change model, notebook, or experiment settings unless the user has authorized that work and the active experiment contract is settled.
- When a task changes behavior, update relevant documentation and run the focused checks required by the affected code. Do not run experiments to validate documentation-only changes.

## Personal workflow integration

Use `route-task` for the shared model/effort and delegation policy and
`master-workflow` for chat, checkout and host placement. Keep project-specific
science, commands and evidence requirements here; do not duplicate personal
model tables or activate an automatic research/code/experiment loop.
Preserve this project's existing context and source-of-truth layout. Before
handoff, identify the actual checkout/ref or snapshot and explicitly transfer
required ignored notes, outputs and environments. A new chat does not itself
authorize a new worktree, experiment, Goal or campaign.
