# Implementation Plan: Reference Authenticity Audit (BMC Public Health)

## Phase 1: Environment & Tooling Setup
*Goal: Prepare the audit workspace and extraction scripts.*

- [ ] Task: Initialize track directory structure and manifest.
    - [ ] Create `conductor/tracks/ref_verification_20260414/reports/`.
    - [ ] Create `conductor/tracks/ref_verification_20260414/metadata.json`.
- [ ] Task: Extract and tokenize references from `manuscript.submission.md`.
    - [ ] Parse Vancouver format into a temporary `audit_master_list.json`.
    - [ ] Verify count (expecting 38 references).
- [ ] Task: Conductor - User Manual Verification 'Phase 1' (Protocol in workflow.md)

## Phase 2: Systematic Verification (Batch 1: Refs 1-10)
*Goal: High-rigor verification of the first 10 citations.*

- [ ] Task: Verify DOIs and metadata for Refs 1-10.
    - [ ] Check DOIs via `curl` and `rg` against CrossRef/PubMed.
    - [ ] Log raw results to `reports/raw/batch_01/`.
- [ ] Task: Analyze discrepancies and categorize status (VERIFIED/DRIFT/BLOCKER).
- [ ] Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)

## Phase 3: Systematic Verification (Batch 2: Refs 11-20)
*Goal: High-rigor verification of citations 11-20.*

- [ ] Task: Verify DOIs and metadata for Refs 11-20.
    - [ ] Query Semantic Scholar/PubMed for each.
- [ ] Task: Analyze discrepancies and categorize status.
- [ ] Task: Conductor - User Manual Verification 'Phase 3' (Protocol in workflow.md)

## Phase 4: Systematic Verification (Batch 3: Refs 21-30)
*Goal: High-rigor verification of citations 21-30.*

- [ ] Task: Verify DOIs and metadata for Refs 21-30.
- [ ] Task: Analyze discrepancies and categorize status.
- [ ] Task: Conductor - User Manual Verification 'Phase 4' (Protocol in workflow.md)

## Phase 5: Systematic Verification (Batch 4: Refs 31-38)
*Goal: High-rigor verification of citations 31-38.*

- [ ] Task: Verify DOIs and metadata for Refs 31-38.
- [ ] Task: Analyze discrepancies and categorize status.
- [ ] Task: Conductor - User Manual Verification 'Phase 5' (Protocol in workflow.md)

## Phase 6: Deep Dive into Suspect References
*Goal: Confirm hallucinations for any flagged items using triple-source verification.*

- [ ] Task: Perform "Penetration Search" for all SUSPECTED_HALLUCINATION entries.
    - [ ] Google/Web search for author profiles and publication lists.
    - [ ] Check for retracted papers or title drift.
- [ ] Task: Finalize evidence logs for any BLOCKER findings.
- [ ] Task: Conductor - User Manual Verification 'Phase 6' (Protocol in workflow.md)

## Phase 7: Synthesis & Audit Pack Generation
*Goal: Produce the final auditable report.*

- [ ] Task: Compile `verification_table.tsv`.
- [ ] Task: Draft `audit_report.md` with summary statistics and high-severity findings.
- [ ] Task: Conductor - User Manual Verification 'Phase 7' (Protocol in workflow.md)
