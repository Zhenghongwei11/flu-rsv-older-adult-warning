# Conductor Tech Stack (Audit Toolchain)

This file describes the **audit toolchain** (not the research project’s tech stack).

## Required Tools (Typical)
- Shell: `bash`/`zsh`
- File inspection: `ls`, `file`, `head`, `sed`, `awk`, `gzip`, `tar`
- Search: `rg` (ripgrep) — always use bounded scans (e.g., `--max-filesize 1M`, `-m 50`) and exclude large dirs per `audit_rules.md` Rule 13
- Hashing: `shasum -a 256` (SHA256)
- HTTP: `curl` (use `-I` for headers; `-L` for redirects; timeouts; explicit proxy control when needed)

## Optional Tools (If Present)
- Python: `python3` (audit scripts are stdlib-only; no third-party deps)
- R: `Rscript` (to read/verify outputs and session info when needed)
- PubMed Search Skill: `pubmed-search` scripts for citation verification.
- Web Search Skill: `web-search` scripts for broad academic verification.
- OCR/PDF tooling if verifying figure regeneration (project-dependent)
- External Tools: `git` (for repo checks), `curl` (for checking DOIs/URLs).

## Security Rules
- Never print secrets (tokens, JWTs, API keys). Redact in reports.
- Prefer reproducible, non-interactive commands.
- If network is blocked (proxy/DNS/401), switch to offline evidence or mark **INCONCLUSIVE**.
- Avoid output storms: redirect full command outputs to `conductor/tracks/<track_id>/reports/` and only preview a small snippet in chat (Rule 13).
