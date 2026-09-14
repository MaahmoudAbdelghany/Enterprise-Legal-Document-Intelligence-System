# LexisGraph — Task Plan

## Phase 1: Foundation & Core Pipeline
- [x] Project scaffolding (pyproject.toml, directory structure, .env.example)
- [x] docker-compose.yml (Qdrant + Neo4j + App)
- [x] app/config.py — centralized settings
- [x] app/core/ingestion/pdf_processor.py — PDF to images
- [x] app/core/ingestion/colpali_embedder.py — ColPali multi-vector embeddings
- [x] app/core/ingestion/text_extractor.py — LLM text extraction
- [x] app/core/retrieval/qdrant_store.py — Qdrant multi-vector CRUD
- [ ] app/core/generation/chain.py — LCEL RAG chain
- [ ] app/core/generation/prompts.py — Arabic-aware prompts
- [ ] app/models/schemas.py — Pydantic models
- [ ] app/api/routes/documents.py — upload/list/delete endpoints
- [ ] app/api/routes/query.py — query endpoint
- [ ] app/api/routes/health.py — health check
- [ ] app/main.py — FastAPI entry point
- [ ] End-to-end test: upload PDF → query → get answer

## Phase 2: Hybrid Retrieval & Knowledge Graph
- [ ] app/core/retrieval/bm25_store.py — BM25 keyword search
- [ ] app/core/retrieval/hybrid_retriever.py — RRF fusion
- [ ] app/core/retrieval/query_router.py — adaptive routing
- [ ] app/core/knowledge_graph/neo4j_client.py — Neo4j connection
- [ ] app/core/knowledge_graph/entity_extractor.py — LLMGraphTransformer
- [ ] app/core/knowledge_graph/graph_retriever.py — Cypher retrieval
- [ ] app/api/routes/graph.py — graph endpoints
- [ ] Integration tests for hybrid retrieval

## Phase 3: Security & Guardrails
- [ ] app/core/security/presidio_service.py — Arabic PII detection/masking
- [ ] app/core/security/nemo_config/ — NeMo Guardrails configuration
- [ ] app/api/middleware/security.py — security middleware
- [ ] Security integration tests

## Phase 4: Evaluation & Adaptive Routing
- [ ] evaluation/ragas_evaluator.py — RAGAS pipeline
- [ ] evaluation/test_dataset.json — Arabic legal test cases
- [ ] evaluation/benchmark.py — benchmark runner
- [ ] scripts/run_eval.py — evaluation script
- [ ] Achieve RAGAS scores ≥ 0.85

## Phase 5: Production AWS Deployment
- [ ] Dockerfile (multi-stage)
- [ ] infrastructure/aws/cloudformation.yml — AWS stack
- [ ] infrastructure/aws/ecs-task-definition.json — ECS config
- [ ] infrastructure/docker/docker-compose.prod.yml — production compose
- [ ] infrastructure/docker/nginx.conf — reverse proxy
- [ ] infrastructure/aws/deploy.sh — deployment script
- [ ] Production smoke tests
- [ ] Load testing
