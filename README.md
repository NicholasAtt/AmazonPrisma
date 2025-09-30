## AmazonPrisma 

### Text Analysis with NER, Sentence Transformers, and BERTopic 🧠

**AmazonPrisma** is a compact, notebook-driven workspace for exploring and structuring e-commerce text (product descriptions, reviews, support notes). It combines three complementary NLP techniques to extract entities, generate semantic embeddings, and surface themes via topic modeling. Its goal is to transform unstructured text into actionable insights for search, catalog enrichment, quality control, and analytics. The implementation and execution of this project are designed specifically for Google Colab.

The **AmazonPrisma.pdf** serves as a narrative companion, providing context, goals, design choices, and discussion of results. The notebooks contain the reproducible, executable steps.

This repository includes three Jupyter notebooks implementing the pipeline:

* `ner.ipynb` — Named Entity Recognition 🔎
* `sentence_transformers.ipynb` — Semantic Embeddings and Similarity 🔗
* `bertopic.ipynb` — Topic Modeling with BERTopic 🧵

Use the PDF for background and rationale, and run the notebooks end-to-end on Google Colab.

---

### Objectives 🎯

* Extract reliable entities (e.g., brands, categories, specifications) from raw text to improve data quality and downstream features.
* Encode sentences/documents into dense vectors for semantic search, deduplication, and matching.
* Discover interpretable topics to summarize large text collections and guide exploration.
* Provide a simple, reproducible workflow that runs on Google Colab.

---

### Dataset and Scope 📦

* **Source:** Amazon Review Polarity Dataset (derived from SNAP), spanning ~18 years up to March 2013.
* **Attributes per review:** `polarity` (1 = negative, 2 = positive), `title`, `text`.
* **Working sample:** 10,000 randomly sampled reviews (subset chosen for fast iteration and validation).
* **Platform:** Google Colab, providing a pre-configured GPU-enabled environment with easy Drive integration.

---

### Repository Structure 📁

* `docs/`
  * `AmazonPrisma.pdf` — Narrative description and results
* `notebooks/`
  * `ner.ipynb` — Named Entity Recognition
  * `sentence_transformers.ipynb` — Sentence embeddings and similarity
  * `bertopic.ipynb` — Topic modeling with BERTopic
* `README.md`

---

### Requirements and Environment 🧰

* Google Colab runtime (recommended GPU when available)
* Dependencies are installed directly within the notebooks (e.g., spaCy/Hugging Face for NER, `sentence-transformers` for embeddings, `bertopic` for topic modeling). Colab setup cells handle installation.

---

### How to Run (Google Colab) ▶️

1. Open Google Colab.
2. Upload a notebook from `notebooks/` (File → Upload notebook) or import from GitHub/Drive.
3. (Optional) Switch runtime to GPU: Runtime → Change runtime type → Hardware accelerator → GPU.
4. Run the setup cells; dependencies will be installed automatically.
5. Execute the notebook top‑to‑bottom to reproduce the analysis.

---

### Developed on Google Colab ☁️

This project was conceived and developed on Google Colab to maximize ease of setup, reproducibility, and access to accelerated hardware.

* Why this approach
  * Zero‑setup: avoids complex local installations and version conflicts; notebooks declare and install what they need.
  * Accelerated hardware: optional GPU/TPU helps Transformer‑based workloads (spaCy transformers, sentence‑transformers, BERTopic embeddings) run faster.
  * Reproducibility: a clean runtime per session ensures consistent starts; notebooks include exact install cells and parameters.
  * Shareability: one‑click access via a browser; collaborators can run the same notebooks without local prep.
  * Drive integration: seamless dataset access and artifact persistence using Google Drive.

* Typical Colab workflow
  1. Open the desired notebook in Colab.
  2. (Optional) Switch runtime to GPU.
  3. Run the setup cells (install spaCy, transformers, sentence‑transformers, BERTopic, UMAP, HDBSCAN, etc.).
  4. Mount Google Drive to access/save data and results:
     
     from google.colab import drive
     drive.mount('/content/drive')
  5. Execute the notebook cells top‑to‑bottom to reproduce the analysis.

* Data access and persistence
  * Store datasets in Drive and load via the mounted path (e.g., `/content/drive/MyDrive/...`).
  * Save outputs (figures, tables, exports) back to Drive so they persist across sessions.

* Model downloads
  * Hugging Face and spaCy models are downloaded at runtime on first use. Colab caches them during the session; persisting to Drive is optional if needed.

---

### Methods at a Glance 📚

#### NER (`ner.ipynb`) 🔎🧠

**Function:** Recognizes and optionally normalizes entities such as brands, categories, locations, amounts, and simple product attributes.
**Typical outputs:** Entity tables with type, offsets, and context; summary counts and example spans. 📊

##### Key details:

* Preprocessing normalizes text (ASCII conversion, punctuation/whitespace cleanup) and combines `title` + `text`.
* Uses spaCy `en_core_web_trf` for high accuracy, leveraging GPU if available.
* Filters relevant labels: {PERSON, ORG, GPE, PRODUCT, WORK_OF_ART, EVENT, LOC}; generic terms are removed.
* Builds an entity-only corpus vectorized with TF-IDF → UMAP (2D) → HDBSCAN; red/noise cluster reflects diffuse or weak themes.

#### Sentence Transformers (`sentence_transformers.ipynb`) 🔗📏

**Function:** Converts sentences/documents into dense embeddings for semantic comparison (similarity, nearest neighbors, light clustering, retrieval).
**Typical outputs:** Embedding matrix, similarity heatmaps, top-k nearest texts for a query. 🔥

##### Key details:

* Text cleaned with `preprocess_text`, then encoded using `all-MiniLM-L6-v2` (384-dim) via `sentence-transformers` (GPU if available).
* Dimensionality reduction via UMAP (e.g., n_neighbors ≈ 40, metric = cosine), clustering with HDBSCAN (min_cluster_size ≈ 15).
* Topic keywords extracted with c-TF-IDF highlight coherent themes (e.g., DVD players — very negative; coffee makers — mixed; printers — negative).
* Applications: semantic search, title ↔ text coherence analysis, cluster inspection.

#### BERTopic (`bertopic.ipynb`) 🧵📊

**Function:** Discovers latent topics using BERTopic (UMAP + HDBSCAN + topic representation) to summarize recurring themes.
**Typical outputs:** Topic list with keywords, document → topic assignments, interactive visualizations. 📈

##### Key details:

* Hybrid enrichment: Transformer-based NER (e.g., `dbmdz/bert-large-cased-finetuned-conll03-english`) + TagMe entity linking.
* Merges overlapping annotations, prioritizes linked entities, adds context (e.g., “iPhone → iPhone product”, “Jobs → Steve Jobs”).
* BERTopic stack: sentence-transformers → UMAP → HDBSCAN → c-TF-IDF.
* Benefits: less generic, clearer, more stable topics; enables polarity, emotion classification, and aspect-based sentiment analysis (ABSA) per topic.

---

### Input Data 📝

The notebooks work with text data (e.g., product descriptions, reviews, support notes). If no sample datasets are provided, you can:

* Upload a CSV/JSON file with a text column.
* Manually insert a list of sentences.
* Connect an external source (e.g., API) if supported in the notebook.

---

### End-to-End Workflow 🧭

1. Read `AmazonPrisma.pdf` for context and rationale.
2. Run `ner.ipynb` to extract entities for cleaning, enrichment, and feature engineering.
3. Run `sentence_transformers.ipynb` to generate embeddings for semantic search, deduplication, and matching.
4. Run `bertopic.ipynb` to discover topics and interpret clusters; use topics to guide exploration.
5. Iterate: refine models and parameters (e.g., NER model, embedding backbone, BERTopic `min_cluster_size`) and re-evaluate.

**Evaluation highlights:**

* Cluster quality measured with Silhouette, Calinski-Harabasz, and Davies-Bouldin scores (excluding noise points as appropriate).
* Polarity analysis: clear separation of negative themes (refunds, faulty devices, damaged packaging) vs positive themes (gifts, hobbies, personal items).
* Emotion analysis (`j-hartmann/emotion-english-distilroberta-base`): sadness dominates negative clusters; joy dominates gift-related clusters.
* ABSA with zero-shot (`facebook/bart-large-mnli`) reveals aspect-level sentiment (e.g., sound/price negative; design/quality sometimes positive).

---

### Typical Use Cases 💼

* Product catalog enrichment: infer brands, categories, and attributes from descriptions.
* Semantic search & retrieval: find similar products or reviews beyond keyword matches.
* Deduplication and matching: detect near-duplicate listings or merge similar SKUs.
* Quality monitoring: identify emerging issues and recurring themes in reviews/support notes.
* Analytics & exploration: summarize large text corpora into interpretable topics for stakeholders.

---

### Quality, Performance, and Reproducibility ⚙️📌

* Results depend on model/language choice and hyperparameters. Log all configurations (model names, seeds, key parameters).
* Evaluate NER via spot‑checks or labeled samples; embeddings via qualitative neighbor inspection or downstream metrics.
* BERTopic quality depends on UMAP/HDBSCAN settings; tune parameters and assess topic coherence.
* Prefer running on Colab for a clean, shareable environment with consistent kernels.

---

### Results Snapshot 📈

* 10k-review subset provided a practical testbed with meaningful patterns.
* Themes: coherent clusters for devices (DVD players, printers), appliances (coffee makers), service topics (refunds, packaging), and lifestyle (music, gifts).
* Sentiment: distinct positive vs negative topic groups; emotion analysis adds nuance.
* Utility: semantic search retrieves intent-aligned matches; title ↔ text coherence highlights mismatched or low-signal titles.

---

### Data Privacy and Ethics 🔐

* Ensure texts do not contain sensitive or personal data; anonymize if necessary.
* Respect content licenses and platform policies when scraping or aggregating text.
* Use models responsibly; verify topics with human review before drawing conclusions.

---

### Roadmap 🗺️

* Add lightweight labeling examples to assess NER precision/recall.
* Provide retrieval demos (semantic search UI in Colab).
* Offer configurable pipelines (YAML/JSON) to save/load settings.
* Optional: export topics/entities to CSV/Parquet for BI tools.
* Scale to larger samples; compare embedding backbones; add coherence metrics; include lightweight human evaluation.

---

### Support 🆘

* Run the notebooks on Google Colab for a zero‑setup experience.
* For methodology or project rationale, consult `AmazonPrisma.pdf`.

---
