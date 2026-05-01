"""Reusable prompt definitions for RAG generation."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate

from rag_experiment.retrieval.bm25 import RetrievalResult


@dataclass(frozen=True)
class PromptDefinition:
    id: str
    description: str
    system_template: str
    human_template: str

    def as_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "description": self.description,
            "system_template": self.system_template,
            "human_template": self.human_template,
        }

    def render(self, *, question: str, context: str) -> list[dict[str, str]]:
        template = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_template),
                ("human", self.human_template),
            ]
        )
        messages = template.format_messages(question=question, context=context)
        return [{"type": message.type, "content": str(message.content)} for message in messages]


RAG_QA_V1 = PromptDefinition(
    id="rag_qa_v1",
    description="Answer a knowledge-intensive question using retrieved passages.",
    system_template=(
        "You are a careful question-answering assistant. Answer using only the "
        "retrieved context. If the context is insufficient, say that the answer "
        "is not supported by the retrieved passages."
    ),
    human_template=(
        "Question:\n{question}\n\n"
        "Retrieved context:\n{context}\n\n"
        "Return a concise answer. Include citations using passage ids in square "
        "brackets, for example [mini_001::Title::0]."
    ),
)


PROMPTS = {
    RAG_QA_V1.id: RAG_QA_V1,
}


def get_prompt(prompt_id: str) -> PromptDefinition:
    try:
        return PROMPTS[prompt_id]
    except KeyError as exc:
        available = ", ".join(sorted(PROMPTS))
        raise ValueError(f"Unknown prompt id {prompt_id!r}. Available prompts: {available}") from exc


def format_retrieved_context(results: list[RetrievalResult]) -> str:
    blocks: list[str] = []
    for result in results:
        passage = result.passage
        blocks.append(
            "\n".join(
                [
                    f"[{result.rank}] passage_id: {passage.id}",
                    f"title: {passage.title}",
                    f"sentence_index: {passage.sentence_index}",
                    f"text: {passage.text}",
                ]
            )
        )
    return "\n\n".join(blocks)
