from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


SYSTEM_PROMPT = """你是一个校园图书馆 RAG 问答助手。
你只能根据给定上下文作答，不要编造没有出现过的信息。
如果上下文不足以支撑结论，直接说明“未找到足够依据”，并建议用户补充书名、作者、ISBN 或分类等信息。
回答要求：
1. 使用中文。
2. 优先给出明确结论，再给出依据。
3. 涉及书名、作者、ISBN、馆藏位置、借阅规则、开放时间时，尽量逐项列出。
4. 不要输出内部推理过程。"""


def format_context(documents: list[Document], max_chars: int = 4500) -> str:
    blocks: list[str] = []
    total = 0
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        header = " | ".join(
            part
            for part in [
                f"馆藏编号={metadata.get('book_id') or '未知'}",
                f"标题={metadata.get('title') or metadata.get('h2') or metadata.get('h1') or '未知'}",
                f"作者={metadata.get('author') or '未知'}",
                f"ISBN={metadata.get('isbn') or '未知'}",
                f"索书号={metadata.get('call_number') or '未知'}",
                f"分类={metadata.get('category') or '未知'}",
                f"主角={metadata.get('main_characters') or '未知'}",
                f"主题词={metadata.get('subjects') or '未知'}",
                f"位置={metadata.get('shelf') or '未知'}",
                f"可借={metadata.get('available_count') if metadata.get('available_count') is not None else '未知'}",
                f"来源={metadata.get('source_name') or metadata.get('source') or '未知'}",
                f"状态={metadata.get('status') or '未知'}",
            ]
        )
        body = document.page_content.strip()
        block = f"[{index}] {header}\n{body}"
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


def build_messages(question: str, context: str, history: list[dict[str, str]]) -> list[Any]:
    history_lines = []
    for turn in history[-6:]:
        role = turn.get("role", "")
        content = turn.get("content", "")
        history_lines.append(f"{role}: {content}")
    history_text = "\n".join(history_lines).strip()
    user_prompt = f"""历史对话:
{history_text or '无'}

检索上下文:
{context or '无'}

用户问题:
{question}
"""
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]


def make_llm(api_key: str, base_url: str, model: str) -> ChatOpenAI | None:
    if not api_key:
        return None
    return ChatOpenAI(api_key=api_key, base_url=base_url, model=model, temperature=0.2)


def fallback_answer(question: str, context: str) -> str:
    if not context.strip():
        return f"未找到足够依据回答“{question}”。请补充书名、作者、ISBN 或更具体的查询条件。"
    lines = [line.strip() for line in context.splitlines() if line.strip()]
    preview = "\n".join(lines[:8])
    return f"当前知识库可用线索如下，但证据还不够完整：\n{preview}\n\n建议继续补充更具体的书名、作者或 ISBN。"


def clarify_answer(question: str, context_docs: list[Document]) -> str:
    context = format_context(context_docs)
    return fallback_answer(question, context)


def generate_answer(
    llm: ChatOpenAI | None,
    question: str,
    context_docs: list[Document],
    history: list[dict[str, str]],
) -> tuple[str, bool]:
    context = format_context(context_docs)
    if llm is None or not context.strip():
        return fallback_answer(question, context), True
    messages = build_messages(question, context, history)
    try:
        response = llm.invoke(messages)
        return str(response.content).strip(), False
    except Exception:
        return fallback_answer(question, context), True
