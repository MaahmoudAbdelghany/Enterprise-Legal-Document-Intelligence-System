# LexisGraph — Enterprise Legal Arabic Document Intelligence System

LexisGraph is an enterprise-grade legal AI system specializing in Arabic document understanding using OCR-free visual parsing (ColPali), multi-vector Qdrant retrieval, Neo4j knowledge graphs, and Presidio/NeMo security guardrails.

## Features
- **Visual Parsing**: End-to-end PDF processing with ColPali v1.3 multi-vector embeddings.
- **Hybrid Retrieval**: Reciprocal Rank Fusion (RRF) combining ColPali dense retrieval, BM25 keyword search, and Neo4j graph traversal.
- **Arabic Legal Intelligence**: Specialized prompt engineering and entity extraction for Arabic legal documents and contracts.
- **Enterprise Security**: PII masking with Presidio and safety rails with NeMo Guardrails.
- **Production Architecture**: Docker Compose local development and AWS ECS deployment.

## Getting Started

### Prerequisites
- Python >= 3.10 or `uv`
- Docker & Docker Compose

### Quickstart
1. Clone the repository and copy `.env.example`:
   ```bash
   cp .env.example .env
   ```
2. Start infrastructure services:
   ```bash
   docker compose up -d qdrant neo4j
   ```
3. Run the FastAPI application:
   ```bash
   uv run uvicorn app.main:app --reload
   ```
