# PeerLens

An AI-assisted peer review system for physics manuscripts.

## Status
Prototype in development. Phase 0.

## Architecture
Multi-agent pipeline built on Claude API:
- Ingestion (PDF → structured text)
- Classification (subdomain + paper type)
- Specialist review agents (math, literature, statistics, reproducibility, ethics, domain)
- Synthesis layer

## Corpus
SciPost Physics (2020-2025), CC-BY licensed with open peer review reports.
