#!/usr/bin/env python3
"""Backend do GBAZP Chat: RAG acadêmico com modelos abertos.

Busca metadados e resumos em OpenAlex (diretório acadêmico aberto) e usa qualquer
servidor compatível com a API OpenAI, como Ollama, vLLM ou LocalAI.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from flask import Flask, jsonify, request, send_from_directory

ROOT = os.path.dirname(os.path.abspath(__file__))
OPENALEX_URL = "https://api.openalex.org/works"
CROSSREF_URL = "https://api.crossref.org/works"
MODEL_BASE_URL = os.getenv("OPEN_MODEL_BASE_URL", "http://localhost:11434/v1").rstrip("/")
MODEL_NAME = os.getenv("OPEN_MODEL", "qwen2.5:7b")
MODEL_API_KEY = os.getenv("OPEN_MODEL_API_KEY", "")
REQUEST_TIMEOUT = float(os.getenv("ACADEMIC_SEARCH_TIMEOUT", "12"))
MAX_SOURCES = 6

app = Flask(__name__, static_folder="GBAZP-CHAT", static_url_path="")


def http_json(url: str, *, headers: dict[str, str] | None = None, data: bytes | None = None) -> dict[str, Any]:
    request_headers = {"User-Agent": "GBAZP-Chat/1.0 (academic retrieval; mailto:contact@example.org)"}
    request_headers.update(headers or {})
    req = Request(url, headers=request_headers, data=data, method="POST" if data else "GET")
    with urlopen(req, timeout=REQUEST_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str:
    if not inverted:
        return ""
    words: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        words.extend((position, word) for position in positions)
    return " ".join(word for _, word in sorted(words))[:2400]


def normalize_openalex(work: dict[str, Any]) -> dict[str, Any] | None:
    title = (work.get("title") or "").strip()
    if not title:
        return None
    locations = work.get("authorships") or []
    institutions: list[str] = []
    for authorship in locations:
        for institution in authorship.get("institutions") or []:
            name = institution.get("display_name")
            if name and name not in institutions:
                institutions.append(name)
    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    url = work.get("doi") or primary.get("landing_page_url") or work.get("id")
    return {
        "title": title,
        "year": work.get("publication_year"),
        "type": work.get("type") or "work",
        "authors": [a.get("author", {}).get("display_name") for a in (work.get("authorships") or [])[:5] if a.get("author", {}).get("display_name")],
        "venue": source.get("display_name") or "Periódico/repositório não informado",
        "institutions": institutions[:4],
        "doi": work.get("doi"),
        "url": url,
        "openalex_url": work.get("id"),
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "cited_by": work.get("cited_by_count", 0),
    }


def search_openalex(query: str) -> list[dict[str, Any]]:
    fields = "id,title,publication_year,type,authorships,primary_location,doi,abstract_inverted_index,cited_by_count"
    url = f"{OPENALEX_URL}?search={quote_plus(query)}&per-page=8&select={fields}"
    data = http_json(url, headers={"Accept": "application/json"})
    return [item for item in (normalize_openalex(work) for work in data.get("results", [])) if item]


def search_crossref(query: str) -> list[dict[str, Any]]:
    url = f"{CROSSREF_URL}?query.bibliographic={quote_plus(query)}&rows=6&select=DOI,title,author,published,container-title,URL,type"
    data = http_json(url, headers={"Accept": "application/json"})
    results: list[dict[str, Any]] = []
    for item in (data.get("message") or {}).get("items", []):
        title = ((item.get("title") or [""])[0]).strip()
        if not title:
            continue
        date = ((item.get("published") or {}).get("date-parts") or [[]])[0]
        results.append({
            "title": title,
            "year": date[0] if date else None,
            "type": item.get("type") or "work",
            "authors": [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in (item.get("author") or [])[:5]],
            "venue": ((item.get("container-title") or [""])[0]) or "Periódico/repositório não informado",
            "institutions": [],
            "doi": f"https://doi.org/{item['DOI']}" if item.get("DOI") else None,
            "url": item.get("URL") or (f"https://doi.org/{item['DOI']}" if item.get("DOI") else None),
            "openalex_url": None,
            "abstract": re.sub(r"<[^>]+>", " ", item.get("abstract") or "")[:2400],
            "cited_by": 0,
        })
    return results


def retrieve_sources(query: str) -> list[dict[str, Any]]:
    try:
        results = search_openalex(query)
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError):
        results = []
    if len(results) < 3:
        try:
            results.extend(search_crossref(query))
        except (HTTPError, URLError, TimeoutError, ValueError, KeyError):
            pass
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in results:
        key = (result.get("doi") or result.get("title", "")).lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(result)
    return unique[:MAX_SOURCES]


def source_context(sources: list[dict[str, Any]]) -> str:
    chunks = []
    for index, source in enumerate(sources, 1):
        authors = ", ".join(source.get("authors") or [])
        institutions = ", ".join(source.get("institutions") or [])
        chunks.append(
            f"[Fonte {index}] {source['title']} | {source.get('year') or 's.d.'} | "
            f"{source.get('venue')} | autores: {authors or 'não informados'} | "
            f"instituições: {institutions or 'não informadas'} | resumo: {source.get('abstract') or 'resumo indisponível'}"
        )
    return "\n".join(chunks)


def call_open_model(messages: list[dict[str, str]]) -> str | None:
    payload = json.dumps({"model": MODEL_NAME, "messages": messages, "temperature": 0.15, "max_tokens": 900}).encode()
    headers = {"Content-Type": "application/json"}
    if MODEL_API_KEY:
        headers["Authorization"] = f"Bearer {MODEL_API_KEY}"
    try:
        data = http_json(f"{MODEL_BASE_URL}/chat/completions", headers=headers, data=payload)
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError):
        return None
    choices = data.get("choices") or []
    if not choices:
        return None
    content = (choices[0].get("message") or {}).get("content")
    return content.strip() if isinstance(content, str) and content.strip() else None


def grounded_fallback(question: str, sources: list[dict[str, Any]]) -> str:
    if not sources:
        return ("Não encontrei trabalhos acadêmicos verificáveis para esta pergunta. "
                "Tente incluir mais termos, uma espécie, método ou período. Não vou inventar uma resposta sem fonte.")
    lines = ["Não consegui conectar ao modelo aberto configurado, então não vou improvisar uma conclusão. Encontrei estas evidências acadêmicas para orientar a leitura:"]
    for index, source in enumerate(sources[:3], 1):
        abstract = source.get("abstract") or "Resumo indisponível; consulte o registro original."
        lines.append(f"\n**[Fonte {index}] {source['title']}** — {abstract[:420]}…")
    lines.append("\nA resposta interpretativa será habilitada quando o endpoint do modelo aberto estiver disponível.")
    return "".join(lines)


def render_sources(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "\n\n**Fontes consultadas:** nenhuma fonte acadêmica retornada."
    lines = ["\n\n---\n**Fontes acadêmicas consultadas**"]
    for index, source in enumerate(sources, 1):
        url = source.get("url") or source.get("openalex_url")
        link = f"[{source['title']}]({url})" if url else source["title"]
        institution = f" · {', '.join(source['institutions'])}" if source.get("institutions") else ""
        lines.append(f"\n{index}. {link} — {source.get('venue')} ({source.get('year') or 's.d.'}){institution}")
        if source.get("openalex_url") and source.get("url") != source.get("openalex_url"):
            lines.append(f" · [registro OpenAlex]({source['openalex_url']})")
    return "".join(lines)


def build_answer(question: str, history: list[dict[str, str]], sources: list[dict[str, Any]]) -> str:
    if not sources:
        return grounded_fallback(question, sources)
    system = (
        "Você é um assistente acadêmico cuidadoso. Responda em português do Brasil. "
        "Use exclusivamente as fontes fornecidas no contexto. Não invente dados, autores, citações, números ou consenso. "
        "Se a evidência for insuficiente, diga claramente. Diferencie achado, interpretação e limitação. "
        "Cite afirmações com [Fonte N], usando apenas os números existentes. Produza uma resposta direta, coerente e legível, "
        "com no máximo 6 parágrafos e uma seção curta de limitações quando necessário."
    )
    recent = [m for m in history[-6:] if m.get("role") in {"user", "assistant"}]
    messages = [{"role": "system", "content": system}]
    messages.extend({"role": m["role"], "content": str(m.get("content", ""))[:2500]} for m in recent)
    messages.append({"role": "user", "content": f"Pergunta atual: {question}\n\nContexto acadêmico recuperado:\n{source_context(sources)}"})
    answer = call_open_model(messages) or grounded_fallback(question, sources)
    return answer + render_sources(sources)


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "model": MODEL_NAME, "model_base_url": MODEL_BASE_URL, "retriever": "OpenAlex + Crossref"})


@app.post("/api/chat")
def chat():
    body = request.get_json(silent=True) or {}
    question = str(body.get("message") or "").strip()
    if not question:
        return jsonify({"error": "Envie uma pergunta."}), 400
    if len(question) > 4000:
        return jsonify({"error": "A pergunta deve ter no máximo 4.000 caracteres."}), 413
    history = body.get("history") if isinstance(body.get("history"), list) else []
    sources = retrieve_sources(question)
    return jsonify({"answer": build_answer(question, history, sources), "sources": sources, "model": MODEL_NAME, "grounded": bool(sources)})


@app.get("/")
def index():
    return send_from_directory(ROOT, "index.html")


@app.get("/<path:filename>")
def frontend_asset(filename: str):
    return send_from_directory(os.path.join(ROOT, "GBAZP-CHAT"), filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=os.getenv("FLASK_DEBUG") == "1")
