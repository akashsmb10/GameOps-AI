# Interview guide: 50 questions about GameOps AI

The answers are specific to this repository. Numbers come from `artifacts/` (demo data is synthetic, so say so).

## A. Project and data
1. **What does the project do?** It cleans game reviews, tags complaint categories and measures sentiment with confidence intervals. It retrieves cited evidence with hybrid RAG. A LangGraph agent answers product questions using five approved, read-only tools. FastAPI serves it, and a Streamlit dashboard shows it.
2. **Why synthetic data?** My build environment couldn't reach Steam, Kaggle or Hugging Face, and Steam review text shouldn't be redistributed. So the demo is a clearly labelled generator, and a Steam connector for local real runs exists and is tested against a fixture.
3. **How do you stop synthetic results from being over-claimed?**
   - Every page and response carries a synthetic badge.
   - Labels are stored separately.
   - The README says the scores are an optimistic proxy.
4. **What data-quality problems do you handle?** Duplicate IDs, blank text, unparseable timestamps, spam, non-English text, negative playtime, and update times earlier than creation. Rows are flagged rather than silently dropped, and every rule is counted in the DQ report.
5. **Why keep the original text?** Embeddings, LLMs and citations need the exact words. The normalised copy is only for hashing and analytics.

## B. NLP
6. **How are complaint categories assigned?** It's multi-label over 9 categories plus "Other". I compared keyword rules, embedding similarity, TF-IDF logistic regression, and TF-IDF + rule features. The hybrid was selected on validation micro-F1.
7. **Why multi-label?** One review can complain about servers *and* monetization. One-vs-rest classifiers with per-category thresholds handle that.
8. **What is the most important evaluation lesson here?** Validation must look like deployment. When I validated on training-like phrasing, TF-IDF scored 0.9997 there but 0.29 on new wording. After giving validation unseen phrasing, selection became honest.
9. **Why did plain TF-IDF fail?** It memorised template n-grams. The rule features add domain vocabulary that generalises to new wording.
10. **Source sentiment vs model sentiment?** Source is the player's recommendation, and it drives every metric. Model sentiment is a text classifier's prediction, shown separately to expose model error.
11. **How good is the sentiment model?** ROC-AUC 0.898. Its threshold was tuned on validation, but on unseen wording it over-predicts negatives (precision 0.51, recall 0.98). That's a calibration shift I report rather than hide.
12. **How does negation work in the rules?** A keyword is ignored if "no", "not", "never", "without", "zero" or "nothing" appears in the two tokens before it. "No crashes so far" therefore doesn't count as a bug.

## C. Embeddings, FAISS and RAG
13. **What are embeddings?** Vectors in which similar meaning means nearby vectors. Here they're L2-normalised, so the inner product equals cosine similarity.
14. **Which embedder did you use?** Sentence-Transformers MiniLM is the default when its weights are available. In the build I used spaCy `en_core_web_md` vectors (averaged GloVe-style vectors) because Hugging Face was blocked. The embedder name is stored with the index.
15. **How does FAISS work here?** `IndexFlatIP` does exact inner-product search over all vectors, which is fine for 12k reviews. Metadata filters use `IDSelectorBatch`, so only allowed IDs are searched. At millions of vectors I'd move to IVF or HNSW approximate indexes.
16. **What is semantic search?** Ranking by meaning (vector similarity) rather than exact words.
17. **Why hybrid retrieval?** Averaged word vectors are weak: dense alone reached precision@5 of 0.44. Adding lexical TF-IDF scores fixes exact terms like "2.1". The fusion mode (sum of scores) was picked on a dev set at 0.74.
18. **What is RAG?** Retrieve relevant records, then answer only from them, with citations. That gives fresh, checkable answers without retraining.
19. **Why RAG instead of fine-tuning?** The facts change daily and must be cited. Fine-tuning changes behaviour or format, not up-to-date knowledge.
20. **How do you chunk?** One review is one chunk. Reviews over 120 words become 80-word windows with a 20-word overlap, and documents are split by section.
21. **How do you handle duplicates in results?** Each review gets a text hash (game + normalised text); results are deduplicated by hash and by source ID.
22. **How does no-answer handling work?** At least 50% of the question's content words must occur in the corpus. A cosine threshold failed because "capital of france" scored 0.93 and "crashes" scored 0.47. The gate refused 6/6 off-topic evaluation questions and answered 23/24 answerable ones.
23. **How do you evaluate RAG?**
   - Precision@5 and hit@5 against generator relevance labels.
   - Documentation hit@3.
   - Citation correctness, groundedness and the unsupported-claim rate.
   - No-answer accuracy.
   All are deterministic checks, with no LLM judge.
24. **What is Recall@K?** The share of relevant items that appear in the top K. I report a capped version, |relevant ∩ topK| / min(K, |relevant|), because each question has hundreds of relevant reviews.
25. **What are your RAG scores?** Precision@5 0.956, citation correctness 1.0, groundedness 0.944, all on synthetic data.

## D. Agent
26. **What is an AI agent?** A system where a model decides which actions or tools to use, observes the results and continues until it can answer.
27. **How is it different from a chatbot?** A chatbot replies from its own weights. This agent plans, calls tools that compute real statistics, inspects their outputs and cites them.
28. **What is LangGraph?** A library for building agents as state graphs: nodes update a typed state, and conditional edges decide what runs next. Mine is guard → plan → tools → synthesize → finalize, with a loop between the LLM and the tools.
29. **What is tool calling?** The LLM receives tool schemas and returns structured calls (a name plus JSON arguments) instead of text. I execute them and return the results as tool messages.
30. **How does your agent choose tools?** With an LLM, the model chooses via `bind_tools`. Without one, a labelled deterministic planner extracts the game, dates, version and categories and maps the intent to tools. Example: "why … after update 2.1" → compare + complaints + search + docs.
31. **What happens when a tool fails?**
    - Arguments are validated by Pydantic.
    - Execution runs with a timeout in a thread pool.
    - Failures become structured errors and warnings.
    - The agent answers from the tools that did succeed; this is tested by monkey-patching a tool to raise or to sleep.
32. **How do you limit the agent?** A maximum of 5 tool calls, a recursion limit, a per-tool timeout, whitelisted tools, and no SQL or code execution path.
33. **How do you prevent hallucinations?**
    - Answers come only from tool outputs.
    - Every citation is checked against returned evidence, and invalid ones are removed.
    - If an LLM answer has no valid citation, a deterministic summary replaces it.
    - Percentages are checked against tool values in evaluation.
34. **What is prompt injection, and how do you defend against it?** It's text that tries to override the system's instructions. My defences:
    - A guard refuses known patterns (e.g. "ignore previous instructions", SQL, requests for secrets) before any tool runs.
    - Review text is passed to the model as untrusted data.
    - Most importantly, the tools are read-only and parameterised, so even a successful injection can't write data or run queries.
35. **How do you avoid causal claims?** Comparisons report p-values and sample sizes. Answers state that timing is an association. The agent also flags when the data contradicts the question's premise ("why did negatives drop" when they rose).
36. **How did you evaluate the agent?**
    - 43 deterministic cases: tool selection, multi-tool reasoning, conflicting evidence, missing data, injection, invalid input, argument validation, failure recovery, and the LLM path with a scripted model.
    - Result: 43/43 after fixing four real bugs.
    - 10 held-out cases, run once: 8/10.
37. **Why use a scripted fake LLM in tests?** It verifies the graph wiring: the model chooses tools, non-approved tools are rejected, fabricated citations are stripped, and limits are enforced. It doesn't measure real model quality, and I say so.

## E. LLMs and fine-tuning
38. **What is an LLM?** A large transformer trained to predict the next token. Instruction tuning makes it follow prompts.
39. **What is a Transformer?** Stacked blocks of self-attention and feed-forward layers, with residual connections and normalisation.
40. **What is attention?** Each token's query is compared with the other tokens' keys: softmax(q·k/√d) weights are applied to the values. A causal mask hides future tokens.
41. **What is LoRA?** The weights W are frozen and a low-rank update ΔW = BA is learned. In my demo only 3.3% of parameters were trainable, and the adapter is a small file.
42. **What did your fine-tuning show?** On a random tiny GPT-2, validation loss fell from 7.60 to 5.96, but category F1 stayed at 0. It verified the pipeline (masking, adapters, before/after evaluation), not model quality. The SmolLM2-135M run is implemented but unverified.

## F. Engineering
43. **How does FastAPI serve the app?** Models, indexes and the read-only DuckDB file load once at startup (lifespan). Pydantic validates requests. Errors map to 422 for bad input and 503 when the service isn't built. Middleware writes JSON logs and telemetry, and the agent runs in a thread pool.
44. **Why is DuckDB read-only?** It guarantees no endpoint or tool can change data. Opening with `read_only=True` is enforced and tested.
45. **What is Docker, and how do you use it?** A container packages the app with its dependencies. My image builds the demo pipeline, runs as non-root and has a health check; compose runs the API, the dashboard and optionally Ollama. The build was not executed in my environment.
46. **What is CI/CD here?** GitHub Actions runs ruff, bandit and pip-audit, then tests and evaluations (uploaded as artifacts), then a Docker build and smoke test. pip-audit found 85 known vulnerabilities in my first pins; I upgraded and re-tested.
47. **What does monitoring track?** Request count, latency percentiles, error rates per endpoint, retrieval latency, tool success rate, failed tools, citation validity, dataset refresh time, model/embedder versions and index age. It's labelled as local demo telemetry.
48. **How fast is it?** Locally on 2 vCPU: about 32 req/s at p95 57 ms with realistic think time, and 114 req/s at saturation with p95 540 ms and 0 failures. That's not production evidence.
49. **How would you scale it?**
    - Stateless API replicas, each with read-only copies of the indexes.
    - An approximate FAISS index or a managed vector database.
    - A warehouse for analytics.
    - Response caching for aggregate queries.
    - Queueing for agent requests, since LLM latency would dominate.
50. **What are the limitations of review analysis?**
    - Reviewers self-select.
    - Review bombing and sarcasm happen.
    - Languages differ.
    - Model categories have measurable errors.
    - Temporal coincidence isn't causation.
    - In this repo specifically: synthetic data, weak embeddings, a rule-based planner, and no real-LLM evaluation.
