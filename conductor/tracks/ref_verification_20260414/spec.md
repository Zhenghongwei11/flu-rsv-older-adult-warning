# Track Specification: Reference Authenticity Audit (BMC Public Health)

## Overview
This track involves a "penetration-style" verification of all 38 references cited in the BMC Public Health submission package (`submission/bmc_public_health/package/manuscript.submission.md`). The goal is to identify hallucinated (fake) references, mismatched DOIs, or metadata errors that could compromise the manuscript's credibility during peer review.

## Functional Requirements

### 1. Reference Extraction
- Extract all 38 references from the `References (Vancouver)` section of `submission/bmc_public_health/package/manuscript.submission.md`.
- Parse each reference into components: Authors, Title, Journal, Year, Volume/Issue, Pages, and DOI.

### 2. Multi-Source Verification
- For each reference, verify its existence and metadata accuracy across:
  - **PubMed/PMC**: Primary source for biomedical literature.
  - **CrossRef/DOI**: Direct verification of DOI strings.
  - **Semantic Scholar**: Verification of citation graph and metadata.
  - **Google/Web Search**: Fallback for non-indexed or pre-print materials.

### 3. DOI Consistency Check
- Validate that the provided DOI in the manuscript points exactly to the paper described in the citation.
- If a DOI is missing, attempt to find and verify the correct DOI.

### 4. Integrity Reporting
- Categorize each reference as:
  - **VERIFIED**: Metadata and DOI match exactly.
  - **MINOR_DRIFT**: Minor metadata typos found, but the paper is real and DOI is correct.
  - **MAJOR_DISCREPANCY**: DOI points to a different paper, or metadata is significantly incorrect.
  - **SUSPECTED_HALLUCINATION**: Reference cannot be found in any database (Potential "fake" reference).
- Provide evidence links (URLs) for every verified reference.

## Non-Functional Requirements
- **Evidence Logging**: Every verification step (command used, database queried, result snippet) must be logged in the track's `reports/` directory.
- **Zero-Trust**: Do not assume any reference is real even if it "looks" professional.

## Acceptance Criteria
- [ ] All 38 references are audited.
- [ ] A "Full Audit Pack" is produced in `conductor/tracks/ref_verification_20260414/reports/` containing:
  - `verification_table.tsv`: Structured list of all references with status and evidence.
  - `audit_report.md`: Detailed summary of findings, highlighting any BLOCKERs (fake references).
- [ ] Any reference marked as `SUSPECTED_HALLUCINATION` or `MAJOR_DISCREPANCY` is verified through at least three independent sources before being finalized.

## Out of Scope
- Fixing the references in the source manuscript (Audit-only).
- Updating the citation style (Vancouver to something else).
