"""Reusable prompt definitions for RAG generation."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from rag_experiment.retrieval.base import RetrievalResult


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

    def render_langchain(self, *, question: str, context: str) -> list[BaseMessage]:
        template = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_template),
                ("human", self.human_template),
            ]
        )
        return template.format_messages(question=question, context=context)

    def render(self, *, question: str, context: str) -> list[dict[str, str]]:
        messages = self.render_langchain(question=question, context=context)
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

RAG_QA_JSON_V1 = PromptDefinition(
    id="rag_qa_json_v1",
    description="Answer with strict JSON containing a short answer and cited passage ids.",
    system_template=(
        "You are a careful question-answering assistant. Answer using only the "
        "retrieved context. Return only valid JSON with exactly these keys: "
        "answer and cited_passage_ids. cited_passage_ids must be a list of "
        "passage_id strings copied from the retrieved context. If the answer is "
        "not supported, say that in answer and return an empty cited_passage_ids list."
    ),
    human_template=(
        "Question:\n{question}\n\n"
        "Retrieved context:\n{context}\n\n"
        "Return only JSON in this schema:\n"
        "{{\"answer\": \"short answer\", \"cited_passage_ids\": [\"passage_id\"]}}"
    ),
)

PUBMEDQA_RAG_JSON_V1 = PromptDefinition(
    id="pubmedqa_rag_json_v1",
    description="Answer PubMedQA using strict yes/no/maybe JSON with citations.",
    system_template=(
        "You are a biomedical question-answering assistant. Answer using only "
        "the retrieved PubMedQA context. Return only valid JSON with exactly "
        "these keys: answer and cited_passage_ids. answer must be exactly one "
        "of: yes, no, maybe. cited_passage_ids must be a list of passage_id "
        "strings copied from the retrieved context. If the retrieved context "
        "does not support a yes or no answer, use maybe."
    ),
    human_template=(
        "Question:\n{question}\n\n"
        "Retrieved context:\n{context}\n\n"
        "Return only JSON in this schema:\n"
        "{{\"answer\": \"yes|no|maybe\", \"cited_passage_ids\": [\"passage_id\"]}}"
    ),
)

PUBMEDQA_RAG_JSON_V2 = PromptDefinition(
    id="pubmedqa_rag_json_v2",
    description="Answer PubMedQA with a less conservative yes/no/maybe policy.",
    system_template=(
        "You are a biomedical question-answering assistant. Answer using only "
        "the retrieved PubMedQA context. Return only valid JSON with exactly "
        "these keys: answer and cited_passage_ids. answer must be exactly one "
        "of: yes, no, maybe. Decide the answer according to the wording of the "
        "question, not according to whether the biomedical outcome is good or "
        "bad. Choose yes when the retrieved abstract supports the claim implied "
        "by the question. Choose no when the retrieved abstract contradicts the "
        "claim implied by the question, reports no meaningful effect for the "
        "intervention or association asked about, or shows failure of the "
        "proposed hypothesis. Choose maybe only when the retrieved evidence is "
        "mixed, explicitly inconclusive, or insufficient to decide yes or no. "
        "Do not use maybe as a default fallback when the abstract states a clear "
        "directional conclusion. cited_passage_ids must be a list of passage_id "
        "strings copied from the retrieved context. Choose cited_passage_ids "
        "first: include only passages that directly support the final answer. "
        "The answer must be the conclusion supported by those cited passages. "
        "If the directly cited passages do not justify a yes or no answer, "
        "choose maybe. Do not cite passages that are merely related but do not "
        "support the answer. Use the smallest set of passage IDs that directly "
        "support the answer."
    ),
    human_template=(
        "Question:\n{question}\n\n"
        "Retrieved context:\n{context}\n\n"
        "Return only JSON in this schema:\n"
        "{{\"answer\": \"yes|no|maybe\", \"cited_passage_ids\": [\"passage_id\"]}}"
    ),
)

PUBMEDQA_RAG_JSON_V3_DEBUG = PromptDefinition(
    id="pubmedqa_rag_json_v3_debug",
    description="Debug PubMedQA generation with citation-first evidence summary.",
    system_template=(
        "You are a biomedical question-answering assistant. Answer using only "
        "the retrieved PubMedQA context. Choose cited_passage_ids first. "
        "Include only passages that directly support the final answer. Then "
        "write a one-sentence evidence_summary that states what the cited "
        "passages conclude. The final answer must be the conclusion supported "
        "by the cited passages and must be exactly one of: yes, no, maybe. "
        "Decide the answer according to the wording of the question, not "
        "according to whether the biomedical outcome is good or bad. If the "
        "cited passages do not clearly justify yes or no, answer maybe. "
        "cited_passage_ids must be passage_id strings copied exactly from the "
        "retrieved context. Do not cite passages that are merely related but do "
        "not support the answer. Return only valid JSON with exactly these "
        "keys: cited_passage_ids, evidence_summary, answer."
    ),
    human_template=(
        "Question:\n{question}\n\n"
        "Retrieved context:\n{context}\n\n"
        "Return only JSON in this schema:\n"
        "{{\"cited_passage_ids\": [\"passage_id\"], "
        "\"evidence_summary\": \"one sentence\", "
        "\"answer\": \"yes|no|maybe\"}}"
    ),
)

SCIFACT_RAG_JSON_V1 = PromptDefinition(
    id="scifact_rag_json_v1",
    description="Classify SciFact claims using strict label JSON with citations.",
    system_template=(
        "You are a scientific claim-verification assistant. Use only the "
        "retrieved SciFact evidence. Return only valid JSON with exactly these "
        "keys: answer and cited_passage_ids. answer must be exactly one of: "
        "SUPPORT, CONTRADICT, NOT_ENOUGH_INFO. cited_passage_ids must be a "
        "list of passage_id strings copied from the retrieved context."
    ),
    human_template=(
        "Claim:\n{question}\n\n"
        "Retrieved evidence:\n{context}\n\n"
        "Return only JSON in this schema:\n"
        "{{\"answer\": \"SUPPORT|CONTRADICT|NOT_ENOUGH_INFO\", "
        "\"cited_passage_ids\": [\"passage_id\"]}}"
    ),
)

SCIFACT_RAG_JSON_V2 = PromptDefinition(
    id="scifact_rag_json_v2",
    description="Classify SciFact claims with stricter support/contradict/NEI policy.",
    system_template=(
        "You are a scientific claim-verification assistant. Use only the "
        "retrieved SciFact evidence. Return only valid JSON with exactly these "
        "keys: answer and cited_passage_ids. answer must be exactly one of: "
        "SUPPORT, CONTRADICT, NOT_ENOUGH_INFO. Choose SUPPORT when the retrieved "
        "evidence directly supports the claim. Choose CONTRADICT when the "
        "retrieved evidence directly refutes or is inconsistent with the claim. "
        "Choose NOT_ENOUGH_INFO only when the retrieved evidence is missing, "
        "irrelevant, or insufficient to decide support or contradiction. Do not "
        "use NOT_ENOUGH_INFO as a default fallback when a retrieved passage gives "
        "direct evidence. cited_passage_ids must be a list of passage_id strings "
        "copied from the retrieved context. Choose cited_passage_ids first: "
        "include only passages that directly support the final label. The label "
        "must be the conclusion supported by those cited passages. If the "
        "directly cited passages do not justify SUPPORT or CONTRADICT, choose "
        "NOT_ENOUGH_INFO. Do not cite passages that are merely related but do "
        "not support the label. Use the smallest set of passage IDs that "
        "directly support the label."
    ),
    human_template=(
        "Claim:\n{question}\n\n"
        "Retrieved evidence:\n{context}\n\n"
        "Return only JSON in this schema:\n"
        "{{\"answer\": \"SUPPORT|CONTRADICT|NOT_ENOUGH_INFO\", "
        "\"cited_passage_ids\": [\"passage_id\"]}}"
    ),
)

SCIFACT_RAG_JSON_V3_DEBUG = PromptDefinition(
    id="scifact_rag_json_v3_debug",
    description="Debug SciFact classification with citation-first evidence summary.",
    system_template=(
        "You are a scientific claim-verification assistant. Use only the "
        "retrieved SciFact evidence. Choose cited_passage_ids first. Include "
        "only passages that directly support the final label. Then write a "
        "one-sentence evidence_summary that states what the cited passages "
        "conclude about the claim. The final answer must be the conclusion "
        "supported by the cited passages and must be exactly one of: SUPPORT, "
        "CONTRADICT, NOT_ENOUGH_INFO. Choose SUPPORT when the cited passages "
        "directly support the claim. Choose CONTRADICT when the cited passages "
        "directly refute or are inconsistent with the claim. If the cited "
        "passages do not clearly justify SUPPORT or CONTRADICT, answer "
        "NOT_ENOUGH_INFO. cited_passage_ids must be passage_id strings copied "
        "exactly from the retrieved context. Do not cite passages that are "
        "merely related but do not support the label. Return only valid JSON "
        "with exactly these keys: cited_passage_ids, evidence_summary, answer."
    ),
    human_template=(
        "Claim:\n{question}\n\n"
        "Retrieved evidence:\n{context}\n\n"
        "Return only JSON in this schema:\n"
        "{{\"cited_passage_ids\": [\"passage_id\"], "
        "\"evidence_summary\": \"one sentence\", "
        "\"answer\": \"SUPPORT|CONTRADICT|NOT_ENOUGH_INFO\"}}"
    ),
)


PROMPTS = {
    RAG_QA_V1.id: RAG_QA_V1,
    RAG_QA_JSON_V1.id: RAG_QA_JSON_V1,
    PUBMEDQA_RAG_JSON_V1.id: PUBMEDQA_RAG_JSON_V1,
    PUBMEDQA_RAG_JSON_V2.id: PUBMEDQA_RAG_JSON_V2,
    PUBMEDQA_RAG_JSON_V3_DEBUG.id: PUBMEDQA_RAG_JSON_V3_DEBUG,
    SCIFACT_RAG_JSON_V1.id: SCIFACT_RAG_JSON_V1,
    SCIFACT_RAG_JSON_V2.id: SCIFACT_RAG_JSON_V2,
    SCIFACT_RAG_JSON_V3_DEBUG.id: SCIFACT_RAG_JSON_V3_DEBUG,
}

DATASET_DEFAULT_PROMPTS = {
    "pubmedqa": PUBMEDQA_RAG_JSON_V1.id,
    "scifact": SCIFACT_RAG_JSON_V1.id,
}


def get_prompt(prompt_id: str) -> PromptDefinition:
    try:
        return PROMPTS[prompt_id]
    except KeyError as exc:
        available = ", ".join(sorted(PROMPTS))
        raise ValueError(f"Unknown prompt id {prompt_id!r}. Available prompts: {available}") from exc


def default_prompt_id_for_dataset(dataset: str) -> str:
    try:
        return DATASET_DEFAULT_PROMPTS[dataset]
    except KeyError as exc:
        available = ", ".join(sorted(DATASET_DEFAULT_PROMPTS))
        raise ValueError(
            f"Unknown dataset {dataset!r}. Available prompt defaults: {available}"
        ) from exc


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


def format_artifact_retrieved_context(passages: list[dict]) -> str:
    """Format normalized artifact retrieved_passages for generation prompts."""
    blocks: list[str] = []
    for index, passage in enumerate(passages, start=1):
        metadata = passage.get("metadata") or {}
        title = (
            passage.get("title")
            or metadata.get("title")
            or metadata.get("context_label")
            or ""
        )
        sentence_index = (
            passage.get("sentence_index")
            if passage.get("sentence_index") is not None
            else metadata.get("sentence_index", metadata.get("context_idx", ""))
        )
        rank = passage.get("rank", index)
        blocks.append(
            "\n".join(
                [
                    f"[{rank}] passage_id: {passage.get('passage_id', '')}",
                    f"title: {title}",
                    f"sentence_index: {sentence_index}",
                    f"text: {passage.get('text', '')}",
                ]
            )
        )
    return "\n\n".join(blocks)
