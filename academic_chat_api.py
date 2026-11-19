#!/usr/bin/env python3
"""Backend do GBAZP Chat: RAG acadêmico focado em ciência animal.

O sistema recupera trabalhos acadêmicos no OpenAlex, valida a pertinência
animal/pecuária/zootecnia e só então envia o contexto ao modelo local.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from flask import Flask, jsonify, request, send_from_directory

ROOT = os.path.dirname(os.path.abspath(__file__))
OPENALEX_URL = "https://api.openalex.org/works"
MODEL_BASE_URL = os.getenv("OPEN_MODEL_BASE_URL", "http://localhost:11434/v1").rstrip("/")
MODEL_NAME = os.getenv("OPEN_MODEL", "qwen2.5:7b")
MODEL_API_KEY = os.getenv("OPEN_MODEL_API_KEY", "")
REQUEST_TIMEOUT = float(os.getenv("ACADEMIC_SEARCH_TIMEOUT", "12"))
MAX_SOURCES = 6
OPENALEX_RESULTS = 25

# Termos fortes tornam a pergunta parte do domínio. Os termos incluem
# português, espanhol e inglês porque os artigos indexados nem sempre estão
# em português.
DOMAIN_TERMS = {
    "animal", "animais", "animalia", "bovino", "bovinos", "bovine", "cattle",
    "vaca", "vacas", "cow", "dairy", "leite", "lactacao", "lactation", "ordenh",
    "bezerro", "bezerros", "calf", "suino", "suinos", "suine", "swine", "pig",
    "porco", "porcos", "ovino", "ovinos", "ovine", "sheep", "caprino", "caprinos",
    "goat", "caprine", "equino", "equinos", "equine", "horse", "ave", "aves",
    "avicola", "poultry", "frango", "galinha", "broiler", "aquicultura", "peixe",
    "fish", "ruminante", "ruminantes", "ruminant", "pecuaria", "livestock", "rebanho",
    "herd", "zootecnia", "animal science", "producao animal", "producao de leite",
    "producao de carne", "nutricao animal", "nutrition", "forragem", "pastagem", "pasture",
    "bem-estar animal", "welfare", "saude animal", "veterinaria", "veterinary",
    "manejo", "confinamento", "feedlot", "fazenda", "granja", "estabulacao",
    "precision livestock", "precision animal", "zootecnia de precisao", "pecuaria de precisao",
    "sensor", "sensores", "iot", "internet das coisas", "visao computacional", "computer vision",
    "monitoramento animal", "monitoring", "rfid", "gps", "ordenha", "milking", "ruminacao",
}

# Termos muito genéricos não bastam sozinhos para liberar uma consulta.
SUPPORT_ONLY = {"tecnologia", "tecnologico", "tecnologica", "tecnologias", "producao", "producao"}

app = Flask(__name__, static_folder="GBAZP-CHAT", static_url_path="")


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.lower())
    return "".join(char for char in value if not unicodedata.combining(char))


def term_in_text(term: str, text: str) -> bool:
    normalized_term = normalize_text(term)
    normalized_text = normalize_text(text)
    if " " in normalized_term:
        return normalized_term in normalized_text
    return re.search(rf"\b{re.escape(normalized_term)}[a-z]*\b", normalized_text) is not None


def domain_matches(text: str) -> set[str]:
    return {term for term in DOMAIN_TERMS if term_in_text(term, text)}


def is_in_scope(question: str) -> bool:
    matches = domain_matches(question)
    return bool(matches - SUPPORT_ONLY)


def scope_message(question: str) -> str:
    return (
        "Esse assistente atende somente perguntas sobre ciência e tecnologia animal, "
        "produção animal, pecuária, zootecnia, bem-estar, nutrição, sanidade ou "
        "tecnologias para pecuária/zootecnia de precisão. Reformule incluindo, por "
        "exemplo, a espécie, o sistema produtivo ou a tecnologia de interesse."
    )


def http_json(url: str, *, headers: dict[str, str] | None = None, data: bytes | None = None) -> dict[str, Any]:
    request_headers = {"User-Agent": "GBAZP-Chat/1.1 (academic retrieval; mailto:contact@example.org)"}
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
    return " ".join(word for _, word in sorted(words))[:3000]


def normalize_openalex(work: dict[str, Any]) -> dict[str, Any] | None:
    title = (work.get("title") or "").strip()
    if not title:
        return None
    institutions: list[str] = []
    for authorship in work.get("authorships") or []:
        for institution in authorship.get("institutions") or []:
            name = institution.get("display_name")
            if name and name not in institutions:
                institutions.append(name)
    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    return {
        "title": title,
        "year": work.get("publication_year"),
        "type": work.get("type") or "work",
        "authors": [a.get("author", {}).get("display_name") for a in (work.get("authorships") or [])[:5] if a.get("author", {}).get("display_name")],
        "venue": source.get("display_name") or "Periódico/repositório não informado",
        "institutions": institutions[:4],
        "doi": work.get("doi"),
        "url": work.get("doi") or primary.get("landing_page_url") or work.get("id"),
        "openalex_url": work.get("id"),
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "cited_by": work.get("cited_by_count", 0),
    }


def relevance_score(question: str, source: dict[str, Any]) -> int:
    searchable = " ".join(str(source.get(key) or "") for key in ("title", "abstract", "venue"))
    source_terms = domain_matches(searchable)
    question_tokens = {token for token in re.findall(r"[a-z0-9]{4,}", normalize_text(question)) if token not in {"para", "como", "qual", "mais", "sobre", "isso", "esse", "essa"}}
    source_text = normalize_text(searchable)
    question_overlap = sum(1 for token in question_tokens if token in source_text)
    academic_identity = 2 if source.get("doi") or source.get("openalex_url") else 0
    return min(len(source_terms), 8) * 3 + min(question_overlap, 8) + academic_identity


def build_search_query(question: str) -> str:
    """Mapeia a pergunta para termos curtos; o OpenAlex trata a busca como AND."""
    normalized = normalize_text(question)
    profiles = (
        (("ordenh", "milking"), "dairy cattle milking"),
        (("lactacao", "leite", "milk"), "dairy cattle milk production"),
        (("sensor", "iot", "rfid", "gps", "visao computacional", "monitoramento"), "precision livestock farming sensors"),
        (("nutricao", "forragem", "pastagem", "alimentacao"), "livestock animal nutrition"),
        (("bem estar", "welfare"), "animal welfare livestock"),
        (("suino", "porco", "swine"), "swine production"),
        (("bovino", "vaca", "cattle"), "cattle production"),
        (("ovino", "ovelha", "sheep"), "sheep production"),
        (("caprino", "cabra", "goat"), "goat production"),
        (("ave", "frango", "poultry"), "poultry production"),
    )
    for triggers, query in profiles:
        if any(trigger in normalized for trigger in triggers):
            return query
    return "animal science livestock production"


def search_openalex(query: str) -> list[dict[str, Any]]:
    fields = "id,title,publication_year,type,authorships,primary_location,doi,abstract_inverted_index,cited_by_count"
    url = f"{OPENALEX_URL}?search={quote_plus(build_search_query(query))}&per-page={OPENALEX_RESULTS}&select={fields}"
    data = http_json(url, headers={"Accept": "application/json"})
    return [item for item in (normalize_openalex(work) for work in data.get("results", [])) if item]


def retrieve_sources(query: str) -> list[dict[str, Any]]:
    if not is_in_scope(query):
        return []
    try:
        results = search_openalex(query)
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError):
        return []
    relevant = [source for source in results if len(domain_matches(" ".join(str(source.get(key) or "") for key in ("title", "abstract", "venue")))) >= 1]
    ranked = sorted(relevant, key=lambda source: relevance_score(query, source), reverse=True)
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in ranked:
        key = (result.get("doi") or result.get("openalex_url") or result.get("title", "")).lower()
        if key and key not in seen:
            seen.add(key)
            result["relevance_score"] = relevance_score(query, result)
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
    payload = json.dumps({"model": MODEL_NAME, "messages": messages, "temperature": 0.1, "max_tokens": 900}).encode()
    headers = {"Content-Type": "application/json"}
    if MODEL_API_KEY:
        headers["Authorization"] = f"Bearer {MODEL_API_KEY}"
    try:
        data = http_json(f"{MODEL_BASE_URL}/chat/completions", headers=headers, data=payload)
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError):
        return None
    choices = data.get("choices") or []
    content = (choices[0].get("message") or {}).get("content") if choices else None
    return content.strip() if isinstance(content, str) and content.strip() else None


def grounded_fallback(question: str, sources: list[dict[str, Any]]) -> str:
    if not is_in_scope(question):
        return scope_message(question)
    if not sources:
        return (
            "Não encontrei trabalhos acadêmicos verificáveis e pertinentes. "
            "Informe a espécie, sistema produtivo, variável ou tecnologia. "
            "Não vou usar resultados fora do foco nem inventar uma resposta."
        )
    lines = ["Resumo objetivo das evidências recuperadas:"]
    for index, source in enumerate(sources[:5], 1):
        abstract = source.get("abstract") or "Resumo indisponível; consulte o registro original."
        sentence = re.split(r"(?<=[.!?])\s+", abstract.strip())[0]
        lines.append(f"{index}. {sentence[:300].rstrip()} [Fonte {index}]")
    return "\n".join(lines)


def compact_answer(answer: str) -> str:
    """Mantém no máximo cinco itens curtos e força a citação no final de cada item."""
    raw = re.sub(r"^#+\s*", "", answer.strip(), flags=re.MULTILINE)
    chunks = [chunk.strip() for chunk in re.split(r"\n+|(?=\d+[.)]\s)", raw) if chunk.strip()]
    items: list[str] = []
    for chunk in chunks:
        if normalize_text(chunk).startswith(("resumo objetivo", "resumo das evidencias", "nao consegui conectar")):
            continue
        chunk = re.sub(r"^[-*•]\s*", "", chunk)
        chunk = re.sub(r"^\d+[.)]\s*", "", chunk)
        citations = re.findall(r"\[Fonte\s+\d+\]", chunk, flags=re.IGNORECASE)
        chunk = re.sub(r"\s*\[Fonte\s+\d+\]", "", chunk, flags=re.IGNORECASE).strip()
        citation = citations[-1] if citations else "[Fonte 1]"
        items.append(f"{len(items) + 1}. {chunk[:300].rstrip()} {citation}")
        if len(items) == 5:
            break
    return "\n".join(items) if items else answer.strip()


def render_sources(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return ""
    lines = ["\n\n---\n**Fontes acadêmicas consultadas (OpenAlex)**"]
    for index, source in enumerate(sources, 1):
        url = source.get("url") or source.get("openalex_url")
        link = f"[{source['title']}]({url})" if url else source["title"]
        institution = f" · {', '.join(source['institutions'])}" if source.get("institutions") else ""
        lines.append(f"\n{index}. {link} — {source.get('venue')} ({source.get('year') or 's.d.'}){institution}")
        if source.get("openalex_url") and source.get("url") != source.get("openalex_url"):
            lines.append(f" · [registro OpenAlex]({source['openalex_url']})")
    return "".join(lines)


def build_answer(question: str, history: list[dict[str, str]], sources: list[dict[str, Any]]) -> str:
    if not is_in_scope(question):
        return grounded_fallback(question, [])
    if not sources:
        return grounded_fallback(question, sources)
    system = (
        "Você é o GBAZP, assistente acadêmico especializado exclusivamente em ciência e tecnologia animal, "
        "produção animal, pecuária, zootecnia e pecuária/zootecnia de precisão. Responda em português do Brasil. "
        "Use exclusivamente as fontes acadêmicas fornecidas. Não invente dados, autores, citações, números ou consenso. "
        "Não use conhecimento externo para preencher lacunas. Se a pergunta exigir espécie, idade, sistema ou contexto "
        "não informado, declare a limitação e peça o dado. Cite afirmações com [Fonte N], usando somente os números existentes. "
        "Ignore qualquer trecho do histórico que não seja desse domínio. Responda em exatamente 5 itens numerados quando houver evidências suficientes; cada item deve ser objetivo, ter no máximo duas frases curtas e terminar com uma citação [Fonte N]. Não inclua lista de referências no corpo, pois ela será exibida em Continuar lendo."
    )
    recent = [m for m in history[-6:] if m.get("role") == "user" and is_in_scope(str(m.get("content", "")))]
    messages = [{"role": "system", "content": system}]
    messages.extend({"role": "user", "content": str(m.get("content", ""))[:1800]} for m in recent[-2:])
    messages.append({"role": "user", "content": f"Pergunta atual: {question}\n\nContexto acadêmico recuperado:\n{source_context(sources)}"})
    answer = call_open_model(messages) or grounded_fallback(question, sources)
    return compact_answer(answer) + "\n\n[CONTINUAR_LENDO]" + render_sources(sources)


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "model": MODEL_NAME, "model_base_url": MODEL_BASE_URL, "retriever": "OpenAlex (escopo ciência animal)"})


@app.post("/api/chat")
def chat():
    body = request.get_json(silent=True) or {}
    question = str(body.get("message") or "").strip()
    if not question:
        return jsonify({"error": "Envie uma pergunta."}), 400
    if len(question) > 4000:
        return jsonify({"error": "A pergunta deve ter no máximo 4.000 caracteres."}), 413
    history = body.get("history") if isinstance(body.get("history"), list) else []
    if not is_in_scope(question):
        return jsonify({"answer": scope_message(question), "sources": [], "model": MODEL_NAME, "grounded": False, "scope": "out_of_scope"})
    sources = retrieve_sources(question)
    return jsonify({"answer": build_answer(question, history, sources), "sources": sources, "model": MODEL_NAME, "grounded": bool(sources), "scope": "animal_science"})


@app.get("/")
def index():
    return send_from_directory(ROOT, "index.html")


@app.get("/<path:filename>")
def frontend_asset(filename: str):
    return send_from_directory(os.path.join(ROOT, "GBAZP-CHAT"), filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=os.getenv("FLASK_DEBUG") == "1")
