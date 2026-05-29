"""Category 8 — Multimodal RAG.

Both recipes here are marked `needs_multimodal=True` so the executor skips
them when the multimodal extras aren't installed. They ship in the repo with
written-out theory, diagrams, and code that runs once the heavy deps are
present.
"""
from __future__ import annotations

from authoring import CodeStep, Recipe, Reference


# =============================================================================
# ColPali — page-as-image retrieval
# =============================================================================
COLPALI = Recipe(
    path="recipes/08-multimodal/colpali-page-as-image.ipynb",
    title="ColPali — Treat PDF Pages as Images, Skip OCR Entirely",
    category="multimodal",
    needs_multimodal=True,

    theory_problem=[
        "PDF retrieval pipelines extract text and lose everything else. Tables collapse into linear reading order. Figures become caption-only references. Equations turn into garbled symbol streams or vanish completely. The expensive parts of the document — the layout, the diagrams, the structured tables — disappear before retrieval ever begins.",
        "ColPali (Faysse et al., 2024) bypasses extraction entirely. Each PDF page is rendered to an image, then embedded by a vision-language model with ColBERT-style late interaction. Retrieval works on patch-level vectors with MaxSim aggregation; queries match the visual structure of pages, not extracted text. State-of-the-art retrieval on PDF benchmarks, no OCR pipeline required, and the model sees figures and tables as first-class content.",
    ],
    theory_origin=[
        "ColPali was published in mid-2024 by Manuel Faysse and colleagues. The paper showed that page-as-image retrieval beat every text-extraction pipeline on the ViDoRe benchmark — by margins as large as 20-30 points on table-heavy and figure-heavy documents. The team open-sourced the model (`vidore/colpali-v1.3`) and `colpali-engine`, the Python library for using it.",
        "By 2026 page-as-image retrieval has become the default for visual PDFs. Variants like ColQwen2.5 and ColNomic-3b push quality further; the underlying pattern is unchanged. The cookbook uses ColPali v1.3 because it's the smallest and easiest to run on a notebook-sized GPU.",
    ],
    theory_landscape=[
        "Three approaches to PDF retrieval in 2026:",
        "",
        "- **Text-extraction RAG.** OCR + text chunking + dense retrieval. Loses structure.",
        "- **ColPali / ColQwen / ColNomic (this recipe).** Render pages, embed with VLM, late interaction. Preserves structure.",
        "- **Hybrid extraction + multimodal.** Extract what extracts well; use multimodal for the rest. More complex.",
        "",
        "Production pipelines mix: text extraction for clean prose, ColPali for tables and figures. The cookbook recipe shows the pure-multimodal path; the hybrid is a composition of recipe 35 plus the text recipes.",
    ],
    theory_when_to_use=[
        "Use ColPali for any corpus where visual structure carries information. Financial filings, scientific papers, slide decks, technical manuals — all good fits because their layout and figures are first-class content.",
        "Skip it for born-digital text (Markdown, HTML, plain text). The visual channel adds nothing and the latency cost is real.",
        "Skip it when you can't run a GPU. CPU inference works but is slow enough to be impractical at notebook scale.",
        "Skip it for very large corpora where storage is the bottleneck. Multi-vector indexes are ~30x bigger than single-vector dense.",
    ],
    theory_intuition=[
        "Four intuitions to carry:",
        "",
        "**Pages are the unit of retrieval.** Not chunks, not tokens — entire rendered pages. The model encodes each page into a grid of patch vectors.",
        "",
        "**Late interaction is the same as ColBERT (Recipe 19).** Query text gets per-token vectors; page gets per-patch vectors; MaxSim aggregates. The same algorithm, different modality.",
        "",
        "**No OCR is the headline feature.** Tables, figures, equations, handwritten notes — everything visible on the page is in the embedding. Nothing is lost to extraction failures.",
        "",
        "**Storage explodes.** Each page is roughly 1024 patch vectors. A 1000-page corpus is a million-vector index. Qdrant and Vespa handle this natively; production usually pairs with Matryoshka (Recipe 12) for cost.",
    ],

    architecture_mermaid="""
flowchart LR
  PDF[PDF document] --> R[Render each page<br/>as PNG]
  R --> V[Vision LM<br/>ColPali v1.3]
  V --> P[Patch vectors<br/>per page]
  P --> S[(Multi-vector store)]
  Q[Query text] --> T[Query token<br/>vectors]
  T --> MS[MaxSim aggregation]
  S --> MS
  MS --> A[Top-k pages]
""",

    references=[
        Reference(
            title="ColPali — Efficient Document Retrieval with Vision Language Models (Faysse et al., 2024)",
            url="https://arxiv.org/abs/2407.01449",
            kind="paper",
            note="The original paper.",
        ),
        Reference(
            title="illuin-tech/colpali repository",
            url="https://github.com/illuin-tech/colpali",
            kind="repo",
            note="Reference implementation and model weights.",
        ),
        Reference(
            title="ViDoRe benchmark",
            url="https://huggingface.co/spaces/vidore/vidore-leaderboard",
            kind="docs",
            note="The leaderboard where page-as-image methods dominate.",
        ),
        Reference(
            title="ColBERT (Recipe 19)",
            url="https://arxiv.org/abs/2004.12832",
            kind="paper",
            note="The text-only ancestor of ColPali.",
        ),
        Reference(
            title="Qdrant multi-vector documentation",
            url="https://qdrant.tech/documentation/concepts/vectors/#multivectors",
            kind="docs",
            note="The native storage path for ColPali in production.",
        ),
        Reference(
            title="ColQwen2.5 / ColNomic-3b alternative checkpoints",
            url="https://huggingface.co/vidore",
            kind="repo",
            note="Other page-as-image embedders.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Render PDF pages to images",
                "",
                "We use the cookbook's arXiv Mamba PDF. PyMuPDF renders each page to a PIL image at ~120 DPI; resolution above that wastes memory without quality gain on text-heavy pages.",
            ],
            code=[
                "# pip install 'rag-cookbook[multimodal]' before running this notebook.",
                "import fitz",
                "from PIL import Image",
                "from io import BytesIO",
                "from pathlib import Path",
                "",
                "PDF = Path('../../corpus/arxiv-2403-mamba.pdf')",
                "doc = fitz.open(PDF)",
                "images = []",
                "for i, page in enumerate(doc):",
                "    pix = page.get_pixmap(dpi=120)",
                "    img = Image.open(BytesIO(pix.tobytes('png'))).convert('RGB')",
                "    images.append((i + 1, img))",
                "    if len(images) >= 12:",
                "        break",
                "print(f'Rendered {len(images)} pages')",
            ],
        ),
        CodeStep(
            tag="embed",
            lead_md=[
                "### Step 2 — Load ColPali",
                "",
                "We load the v1.3 checkpoint from Hugging Face. ColPali takes a few seconds to load and ~4 GB of VRAM on a modest GPU; CPU works but is roughly 100x slower.",
            ],
            code=[
                "import torch",
                "from colpali_engine.models import ColPali, ColPaliProcessor",
                "",
                "device = 'cuda' if torch.cuda.is_available() else 'cpu'",
                "model = ColPali.from_pretrained('vidore/colpali-v1.3', torch_dtype=torch.float32).to(device).eval()",
                "processor = ColPaliProcessor.from_pretrained('vidore/colpali-v1.3')",
                "print(f'ColPali loaded on {device}.')",
            ],
        ),
        CodeStep(
            tag="embed",
            lead_md=[
                "### Step 3 — Embed pages and a query",
                "",
                "Pages get per-patch vectors; queries get per-token vectors. The processor's `score_multi_vector` runs the MaxSim aggregation.",
            ],
            code=[
                "with torch.no_grad():",
                "    page_inputs = processor.process_images([img for _, img in images]).to(device)",
                "    page_embs = model(**page_inputs)",
                "    print(f'Page embeddings: {page_embs.shape}')",
                "",
                "    q_inputs = processor.process_queries(['What is selective scan and why does it matter?']).to(device)",
                "    q_embs = model(**q_inputs)",
                "    print(f'Query embeddings: {q_embs.shape}')",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 4 — Score with MaxSim",
                "",
                "The score is a sum of per-query-token max similarities. Each query token finds its best matching page patch and contributes that match to the total.",
            ],
            code=[
                "with torch.no_grad():",
                "    scores = processor.score_multi_vector(q_embs, page_embs).cpu().tolist()[0]",
                "",
                "ranked = sorted(zip(scores, [n for n, _ in images]), reverse=True)",
                "for s, n in ranked[:5]:",
                "    print(f'  page {n}  score {s:.2f}')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 5 — Wrap as `answer_question`",
                "",
                "The cookbook contract returns (answer, contexts). For ColPali, contexts are descriptions of the top-k pages — recipe 36 shows how to feed the images themselves to a VLM for the answer.",
            ],
            code=[
                "def answer_question(question: str) -> tuple[str, list[str]]:",
                "    with torch.no_grad():",
                "        qi = processor.process_queries([question]).to(device)",
                "        qe = model(**qi)",
                "        sc = processor.score_multi_vector(qe, page_embs).cpu().tolist()[0]",
                "    best = sorted(zip(sc, [n for n, _ in images]), reverse=True)[:3]",
                "    pages = [n for _, n in best]",
                "    return f'Top pages for visual retrieval: {pages}', [f'page {p}' for p in pages]",
                "",
                "ans, _ = answer_question('How does Mamba's parallel scan work?')",
                "print(ans)",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 6 — Visualise the top retrieved page",
                "",
                "Display the highest-scoring page image so you can see what ColPali matched on.",
            ],
            code=[
                "from IPython.display import display",
                "best_idx = ranked[0][1] - 1   # back to 0-based",
                "display(images[best_idx][1].resize((400, 600)))",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — score distribution across all pages",
                "",
                "How sharp is the ranking? A steep curve means ColPali confidently identified the right pages.",
            ],
            code=[
                "import matplotlib.pyplot as plt",
                "fig, ax = plt.subplots(figsize=(6, 2.5))",
                "ax.bar(range(1, len(scores) + 1), scores)",
                "ax.set_xlabel('page')",
                "ax.set_ylabel('MaxSim score')",
                "ax.set_title('ColPali score distribution')",
                "plt.tight_layout()",
                "plt.show()",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — multiple queries, multiple winners",
                "",
                "Different queries should pick different pages. If everything routes to page 1, the model isn't differentiating.",
            ],
            code=[
                "for q in [",
                "    'What is HiPPO initialization?',",
                "    'Show the experimental results.',",
                "    'What is the abstract about?',",
                "    'What does the parallel scan diagram look like?',",
                "]:",
                "    with torch.no_grad():",
                "        qi = processor.process_queries([q]).to(device)",
                "        qe = model(**qi)",
                "        sc = processor.score_multi_vector(qe, page_embs).cpu().tolist()[0]",
                "    best_page = sorted(zip(sc, [n for n, _ in images]), reverse=True)[0][1]",
                "    print(f'  {q[:55]:55s} -> page {best_page}')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — storage cost",
                "",
                "Multi-vector storage is ~30x single-vector. Quantify on the current page set.",
            ],
            code=[
                "single_vec_bytes = 1024 * 4   # typical text embedding",
                "page_size = page_embs.shape[1] * page_embs.shape[2] * 4",
                "print(f'Single-vector per page:    {single_vec_bytes:,} bytes')",
                "print(f'ColPali multi-vector page: {page_size:,} bytes')",
                "print(f'Ratio: ~{page_size // single_vec_bytes}x')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — latency",
                "",
                "Per-page embedding and per-query scoring; measure both.",
            ],
            code=[
                "import time",
                "with torch.no_grad():",
                "    t0 = time.perf_counter()",
                "    _ = model(**processor.process_queries(['probe']).to(device))",
                "    q_ms = (time.perf_counter() - t0) * 1000",
                "    t0 = time.perf_counter()",
                "    _ = processor.score_multi_vector(q_embs, page_embs)",
                "    score_ms = (time.perf_counter() - t0) * 1000",
                "print(f'Query embed: {q_ms:.1f} ms')",
                "print(f'MaxSim score: {score_ms:.1f} ms')",
            ],
        ),
    ],

    run_md=[
        "End-to-end ColPali retrieval.",
    ],
    run_code=[
        "ans, _ = answer_question('Where does the paper show experimental complexity results?')",
        "print(ans)",
    ],

    comparison_md=[
        "Text-only baseline vs ColPali. The interesting cases are pages with tables or figures that text extraction would mangle.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = 'Where does the paper show experimental complexity results?'",
        "base = vanilla_pipeline(q, corpus='arxiv-mamba', top_k=3)",
        "ours_a, _ = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'text-extraction', 'preview': base.answer[:140]},",
        "    {'pipeline': 'colpali', 'preview': ours_a[:140]},",
        "])",
    ],

    tuning_md=[
        "Six knobs in priority order:",
        "",
        "1. **Model checkpoint.** ColPali v1.3, ColQwen2.5, ColNomic-3b are all strong. Benchmark on your corpus.",
        "2. **Render DPI.** 120 is the cookbook default. Higher costs memory and storage; lower loses detail in dense text.",
        "3. **Page count cap.** Notebook demos cap at 12; production indexes thousands of pages per document. Plan for storage.",
        "4. **GPU vs CPU.** CPU works for tiny corpora; GPU is essentially required at scale. A modest 16 GB GPU handles ColPali v1.3 comfortably.",
        "5. **Native multi-vector store.** Qdrant, Vespa, Weaviate all have native multi-vector paths. Use them in production.",
        "6. **Quantisation.** Combine with Matryoshka (Recipe 12) or product quantisation to manage the multi-vector storage cost at scale.",
    ],

    discussion_md=[
        "Four failure modes:",
        "",
        "- **Memory limits.** ColPali on a small GPU runs out of VRAM with 50+ pages in a batch. Stream and process in chunks.",
        "- **Slow CPU inference.** A 25-page document takes 5+ minutes on CPU. Use GPU or accept the wait.",
        "- **No native answer generation.** ColPali retrieves; it does not synthesise. Pair with recipe 36 (VLM synthesis) for end-to-end visual QA.",
        "- **Storage at scale.** Multi-vector indexes are large. Combine with Matryoshka (Recipe 12) or product quantisation to manage cost.",
        "",
        "Compose with recipe 36 (VLM synthesis over pages) for full end-to-end visual RAG, with ColBERT (Recipe 19) for the text-mode cousin, and with hybrid retrieval (Recipe 18) for corpora that mix visual and prose documents.",
    ],
)


# =============================================================================
# VLM Synthesis Over Pages
# =============================================================================
VLM_SYNTH = Recipe(
    path="recipes/08-multimodal/vlm-synthesis-over-pages.ipynb",
    title="VLM Synthesis — Answering from Page Images",
    category="multimodal",
    needs_multimodal=True,

    theory_problem=[
        "ColPali retrieves pages but does not answer questions. You still need to generate an answer, and the answer should reflect what's visible on the page — including the tables and figures the visual retrieval surfaced. Text-only LLMs can't see images; the synthesis step has to use a vision-language model.",
        "Modern VLMs (Qwen2.5-VL, GPT-4o, Gemini 2.5, Llama-3.2-Vision) accept image inputs alongside text. The pattern: ColPali picks the relevant page images, send them plus the question to a VLM, get the answer. The VLM sees the tables, figures, and layout that text extraction would have lost.",
    ],
    theory_origin=[
        "The pattern emerged organically as VLMs became production-capable in 2024. GPT-4V (later GPT-4o) was the first frontier VLM available via API; Gemini Pro Vision followed. By mid-2025, open VLMs like Qwen2.5-VL and Llama-3.2-Vision had reached comparable quality. Combining them with ColPali-style retrieval (the upstream paper coined this composition) became the standard for visual document QA.",
        "By 2026 the composition is so common it has a name in production literature — \"visual RAG\" — and most multimodal RAG demos run this exact pattern. The cookbook factors retrieval (Recipe 35) and synthesis (this recipe) as separate notebooks so each step is individually inspectable, but in production they live as one pipeline.",
    ],
    theory_landscape=[
        "Three approaches to visual document QA in 2026:",
        "",
        "- **Text-extraction + text LLM.** Extracts text, answers from text. Loses visual content.",
        "- **VLM-only.** Send the document straight to the VLM. Works for short documents; doesn't scale.",
        "- **ColPali + VLM (this recipe).** Retrieve pages with ColPali, answer with VLM. Scales and preserves visual content.",
        "",
        "The third pattern is the canonical production setup. The cookbook isolates the retrieval (Recipe 35) and synthesis (Recipe 36) steps so the parts are individually inspectable.",
    ],
    theory_when_to_use=[
        "Use VLM synthesis whenever your retrieved evidence is visual. ColPali outputs page images, financial documents, slide decks, scientific papers — all benefit from visual-aware answering.",
        "Skip it when your retrieval is text-only. There's nothing visual to synthesise from.",
        "Skip it when latency or cost matters more than visual fidelity. VLM calls are slower and more expensive than text-only calls because images cost a lot of input tokens.",
    ],
    theory_intuition=[
        "Four intuitions:",
        "",
        "**Image tokens are expensive.** A 1024×1024 image is roughly 1500-4000 input tokens depending on the VLM. Five pages × 1500 tokens × $5/M input tokens = real money at scale.",
        "",
        "**The VLM sees layout.** Tables, figures, equations, and spatial relationships all live in the image. The VLM uses them directly.",
        "",
        "**Pick few pages, not many.** Sending 50 page images to the VLM is wasteful. ColPali picks the top-3 confidently; that's what the VLM should see.",
        "",
        "**Prompt for grounding.** \"Answer only from the visible content\" reduces VLM hallucination noticeably. The model is otherwise tempted to reason beyond the image and import outside facts that may not be visible.",
    ],

    architecture_mermaid="""
flowchart LR
  CP[ColPali<br/>top-3 pages] --> IMG[Page images]
  Q[Question] --> VLM
  IMG --> VLM[Vision LM]
  VLM --> A[Grounded answer]
""",

    references=[
        Reference(
            title="ColPali (Recipe 35)",
            url="https://arxiv.org/abs/2407.01449",
            kind="paper",
            note="The retrieval step upstream.",
        ),
        Reference(
            title="Qwen2.5-VL model card",
            url="https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct",
            kind="repo",
            note="The open-weight VLM we recommend.",
        ),
        Reference(
            title="OpenAI vision documentation",
            url="https://platform.openai.com/docs/guides/vision",
            kind="docs",
            note="GPT-4o vision API.",
        ),
        Reference(
            title="Anthropic vision documentation",
            url="https://docs.anthropic.com/en/docs/build-with-claude/vision",
            kind="docs",
            note="Claude vision API.",
        ),
        Reference(
            title="Gemini vision documentation",
            url="https://ai.google.dev/gemini-api/docs/vision",
            kind="docs",
            note="Gemini vision API.",
        ),
        Reference(
            title="LiteLLM multimodal documentation",
            url="https://docs.litellm.ai/docs/completion/vision",
            kind="docs",
            note="Cross-provider multimodal calls via LiteLLM.",
        ),
    ],

    cells=[
        CodeStep(
            tag="load",
            lead_md=[
                "### Step 1 — Reuse ColPali retrieval",
                "",
                "This recipe builds on Recipe 35. The setup below reproduces ColPali retrieval; in production you'd reuse a shared state.",
            ],
            code=[
                "# pip install 'rag-cookbook[multimodal]' before running.",
                "import fitz, torch",
                "from PIL import Image",
                "from io import BytesIO",
                "from pathlib import Path",
                "from colpali_engine.models import ColPali, ColPaliProcessor",
                "",
                "PDF = Path('../../corpus/arxiv-2403-mamba.pdf')",
                "doc = fitz.open(PDF)",
                "images = []",
                "for i, page in enumerate(doc):",
                "    pix = page.get_pixmap(dpi=120)",
                "    img = Image.open(BytesIO(pix.tobytes('png'))).convert('RGB')",
                "    images.append((i + 1, img))",
                "    if len(images) >= 12:",
                "        break",
                "",
                "device = 'cuda' if torch.cuda.is_available() else 'cpu'",
                "model = ColPali.from_pretrained('vidore/colpali-v1.3', torch_dtype=torch.float32).to(device).eval()",
                "processor = ColPaliProcessor.from_pretrained('vidore/colpali-v1.3')",
                "",
                "with torch.no_grad():",
                "    page_embs = model(**processor.process_images([img for _, img in images]).to(device))",
                "print(f'ColPali ready: {len(images)} pages indexed.')",
            ],
        ),
        CodeStep(
            tag="retrieve",
            lead_md=[
                "### Step 2 — Pick top-k pages for a query",
                "",
                "Same retrieval as recipe 35. We return PIL images for the top-k pages.",
            ],
            code=[
                "def colpali_pick(question: str, k: int = 3):",
                "    with torch.no_grad():",
                "        qe = model(**processor.process_queries([question]).to(device))",
                "        sc = processor.score_multi_vector(qe, page_embs).cpu().tolist()[0]",
                "    ranked = sorted(zip(sc, range(len(images))), reverse=True)[:k]",
                "    return [images[idx][1] for _, idx in ranked]",
                "",
                "picks = colpali_pick('What is selective scan?')",
                "print(f'Picked {len(picks)} pages.')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 3 — Encode images for the VLM",
                "",
                "We base64-encode each PIL image. The LiteLLM-compatible message format accepts `image_url` with a `data:image/png;base64,...` URL.",
            ],
            code=[
                "import base64, io",
                "",
                "def encode(img: Image.Image) -> str:",
                "    buf = io.BytesIO()",
                "    img.save(buf, format='PNG')",
                "    return base64.b64encode(buf.getvalue()).decode()",
                "",
                "encoded = [encode(img) for img in picks[:2]]",
                "print(f'Encoded {len(encoded)} pages, total size: ~{sum(len(e) for e in encoded) // 1024} KB')",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 4 — Send to the VLM",
                "",
                "Build a multimodal message: one text part with the question, one image part per picked page. The cookbook's `client.chat` supports a list-of-dicts payload directly.",
            ],
            code=[
                "def answer_visually(question: str, pages: list[Image.Image]) -> str:",
                "    parts = [{'type': 'text', 'text': f'Answer using only what is visible in these pages.\\nQ: {question}'}]",
                "    for img in pages:",
                "        parts.append({",
                "            'type': 'image_url',",
                "            'image_url': {'url': f'data:image/png;base64,{encode(img)}'},",
                "        })",
                "    return client.chat([{'role': 'user', 'content': parts}])",
                "",
                "ans = answer_visually('What is selective scan and why does it matter?', picks[:2])",
                "print(ans[:400])",
            ],
        ),
        CodeStep(
            tag="generate",
            lead_md=[
                "### Step 5 — Wrap as `answer_question`",
                "",
                "Cookbook contract. Internally we ColPali-pick then VLM-synthesise.",
            ],
            code=[
                "def answer_question(question: str) -> tuple[str, list[str]]:",
                "    pages = colpali_pick(question, k=3)",
                "    return answer_visually(question, pages), [f'page-image-{i}' for i in range(len(pages))]",
                "",
                "ans, _ = answer_question('Explain the parallel scan implementation.')",
                "print(ans[:400])",
            ],
        ),
    ],

    inspection_cells=[
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — what does the VLM see?",
                "",
                "Display the top page image alongside the answer so you can verify the model is reading from the right source.",
            ],
            code=[
                "from IPython.display import display",
                "display(picks[0].resize((400, 600)))",
                "print('Top page above; VLM answered using this image as context.')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — image token cost",
                "",
                "Each image is roughly 1500-4000 input tokens. Multiply by k pages for the per-query cost.",
            ],
            code=[
                "import sys",
                "encoded_size = sum(sys.getsizeof(e) for e in encoded)",
                "approx_tokens = encoded_size // 3   # base64 inflates ~33%; rough estimate",
                "print(f'Encoded payload: {encoded_size:,} bytes')",
                "print(f'Approximate input tokens: {approx_tokens:,}')",
                "print('Tune k carefully to manage cost.')",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — same question, different page counts",
                "",
                "Compare answers from k=1, k=3, k=5 pages. More pages give the VLM more context but cost more.",
            ],
            code=[
                "q = 'What is selective scan?'",
                "for k in (1, 3, 5):",
                "    pages = colpali_pick(q, k=k)",
                "    ans = answer_visually(q, pages)",
                "    print(f'k={k}: {ans[:200]}')",
                "    print()",
            ],
        ),
        CodeStep(
            tag="inspect",
            lead_md=[
                "### Inspect — out-of-document question",
                "",
                "Ask something the document doesn't cover. The VLM should refuse or hedge — \"Answer only from the visible content\" enforces this.",
            ],
            code=[
                "ans = answer_visually('Who painted the Sistine Chapel ceiling?', picks[:2])",
                "print(ans[:300])",
            ],
        ),
    ],

    run_md=[
        "End-to-end visual QA.",
    ],
    run_code=[
        "ans, _ = answer_question('How does Mamba's complexity compare to attention?')",
        "print('=== VLM-synthesised answer ===')",
        "print(ans)",
    ],

    comparison_md=[
        "Text-extraction baseline vs ColPali + VLM. The contrast is clearest on questions whose answers live in figures or tables.",
    ],
    comparison_code=[
        "from cookbook.baselines import vanilla_pipeline",
        "",
        "q = \"How does Mamba's complexity compare to attention?\"",
        "base = vanilla_pipeline(q, corpus='arxiv-mamba', top_k=3)",
        "ours_a, _ = answer_question(q)",
        "",
        "import pandas as pd",
        "pd.DataFrame([",
        "    {'pipeline': 'text-extraction', 'preview': base.answer[:160]},",
        "    {'pipeline': 'colpali + vlm', 'preview': ours_a[:160]},",
        "])",
    ],

    tuning_md=[
        "Six knobs in priority order:",
        "",
        "1. **VLM choice.** Qwen2.5-VL, GPT-4o, Gemini 2.5, Claude 3.5 Sonnet. Benchmark on your domain; image-handling varies a lot.",
        "2. **Page count `k`.** Default 3. Higher catches more context, costs more tokens. Most queries are well-served by 2-3 pages.",
        "3. **Image resolution.** 120 DPI is the cookbook default; lower if cost is tight, higher if fine details matter for the question.",
        "4. **Prompt grounding.** \"Answer only from the visible content\" reduces hallucination noticeably.",
        "5. **Compose with text RAG.** For born-digital documents, text RAG is cheaper; route per document type.",
        "6. **Batch the VLM calls.** When you have many similar queries against the same pages, batch them together to amortise the image-token cost.",
    ],

    discussion_md=[
        "Three failure modes:",
        "",
        "- **Cost explosion.** Every page image is 1500-4000 input tokens. A k=10 query is roughly 30k input tokens, repeatedly. Plan budgets.",
        "- **VLM hallucination on weak images.** Low-resolution pages or page images with little text confuse VLMs. Grade your inputs.",
        "- **Provider differences.** Image-handling quality varies a lot between VLMs. Test on your domain before committing.",
        "",
        "Compose with ColPali (Recipe 35) for retrieval. Compose with text retrieval recipes for hybrid mixed-modality corpora. Compose with hallucination guardrails (Recipe 40) for the production safety net.",
    ],
)


RECIPES = [COLPALI, VLM_SYNTH]
