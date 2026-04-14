import fitz  # PyMuPDF
from rank_bm25 import BM25Okapi
from langchain_community.chat_models import ChatOllama

# ---------------- INIT LLM (LOCAL) ----------------
llm = ChatOllama(
    model="tinyllama",
    temperature=0.3
)

# ---------------- EXTRACT TEXT ----------------
def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({
                "page": page_num,
                "content": text
            })

    return pages


# ---------------- BUILD STRUCTURE (NO CHUNKING) ----------------
def build_structure(pages):
    """
    Each page = one node (NO chunking)
    """
    tree = []

    for page in pages:
        tree.append({
            "id": page["page"],
            "content": page["content"]
        })

    return tree


# ---------------- BM25 RETRIEVAL ----------------
def build_bm25(tree):
    corpus = [node["content"].split() for node in tree]
    bm25 = BM25Okapi(corpus)
    return bm25


# ---------------- RETRIEVE ----------------
def retrieve(query, tree, bm25, top_k=3):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []
    for idx in ranked_indices[:top_k]:
        results.append(tree[idx]["content"])

    return results


# ---------------- UPLOAD + INDEX ----------------
def upload_and_index(pdf_path):
    pages = extract_text(pdf_path)
    tree = build_structure(pages)
    bm25 = build_bm25(tree)

    return tree, bm25


# ---------------- RAG ----------------
def vectorless_rag(query, tree, bm25):

    contexts = retrieve(query, tree, bm25)

    if not contexts:
        return "No relevant context found."

    combined_context = "\n\n".join(contexts)

    prompt = f"""
Answer clearly and concisely using ONLY the context.

Context:
{combined_context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content 