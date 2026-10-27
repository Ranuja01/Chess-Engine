# dev_notes: how the engine's documentation is organised

Read this before opening anything else here. It says which documents describe the engine as it is, which are work in
progress, and which are history, so you do not mistake an old plan for the current design.

## Folders

| folder | what is in it | how far to trust it |
|---|---|---|
| [`current/`](current/) | Documents that describe the engine as it is: subsystem descriptions, the ship record (`OPTIMIZATION_LOG`), the measuring tools and their limits, the baselines, the live configuration and parked registers. | True as of its **last updated** date (index below). Compare that date with the latest ship in `OPTIMIZATION_LOG` before relying on details. Some describe v1, which is now the legacy control: see the note column. |
| [`active/`](active/) | Plans and research for the arc in progress, and the latest session handoff. | Work in progress; decisions may still change. |
| [`reference/`](reference/) | Write-ups of other engines (Stockfish 11 to 18, Ethereal, Caissa, Obsidian) and techniques taken from them. | About those engines, for the versions they cite; not about ours. |
| [`archive/<topic>/`](archive/) | Superseded plans, one-off investigations, old handoffs. Topics: `handoffs`, `eval`, `king-safety`, `pawns`, `search`, `speed`, `collapse`, `games`, `planning`. | Provenance only ("why did we conclude X"), never current state. See [`archive/README.md`](archive/README.md). |
| [`owner_games/`](owner_games/) | PGNs of the owner's games against the engine. | Data. |

## Where to start

1. [`current/ENGINE-ORIENTATION.md`](current/ENGINE-ORIENTATION.md): file map, data flow, what the search already has.
2. [`current/OPTIMIZATION_LOG.md`](current/OPTIMIZATION_LOG.md): every ship and measured result, newest first.
3. [`current/INSTRUMENT-MAP.md`](current/INSTRUMENT-MAP.md): which measurement answers which question, and how each one misleads.
4. [`current/DIAGNOSTICS-TOOLKIT.md`](current/DIAGNOSTICS-TOOLKIT.md): the probes that already exist. Check here before writing a new one.
5. The latest handoff in [`active/`](active/): where the work stands.
6. For a subsystem: its document in `current/` (time manager, eval v2 configuration, baselines, ...).

## Rules for keeping it tidy

- **Changing a subsystem? Update its `current/` document in the same commit.** A stale description reads as authoritative.
- **New research or a plan** goes in `active/`, with the date in the file name (`TOPIC-YYYY-MM-DD.md`).
- **When an arc closes**, move its `active/` files to `archive/<topic>/`, and promote any lasting finding into a
  `current/` document first. Move older handoffs to `archive/handoffs/`, keeping only the latest in `active/`.
- **After moving files**, rewrite every link (code comments, scripts, memory and other docs reference these paths),
  commit with a subject starting `dev_notes reorganisation` (the index ignores those commits for its dates), and
  regenerate the index: `python dev_notes/_make_index.py`.

## Index

<!-- index:begin -->
<!-- index:end -->
