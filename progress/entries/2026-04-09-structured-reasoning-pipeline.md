# Structured reasoning pipeline

**Summary:** Implemented the staged pipeline from the plan: concept KB loader (`LING487_STRUCTURED_PIPELINE_KB.json`), `StructuredQuery` + decomposition, `AnswerPlan`, rule-based structured Course Answer generation, optional plan-constrained LLM (`LLM_ANSWER_GENERATION`), `validate_answer`, `run_reasoning_pipeline` integrated into `chat_orchestrator` when `STRUCTURED_PIPELINE_ENABLED`, `RetrievalLog` migration `004` for pipeline diagnostics, and `test_structured_pipeline.py`.

**Follow-ups:** Wire study routes into `run_reasoning_pipeline` when desired; tune validation thresholds; expand KB coverage as the corpus grows.
