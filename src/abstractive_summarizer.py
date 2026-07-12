"""Transformer-based abstractive summarization."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .extractive_summarizer import split_sentences
from .utils import compression_ratio, word_count


@dataclass
class GenerationConfig:
    """Generation settings for deterministic inference."""

    max_input_length: int
    max_summary_length: int
    min_summary_length: int
    num_beams: int
    no_repeat_ngram_size: int
    batch_size: int


class AbstractiveSummarizer:
    """Load one seq2seq model and summarize texts safely."""

    def __init__(self, model_name: str, generation_config: GenerationConfig) -> None:
        self.model_name = model_name
        self.config = generation_config
        self.tokenizer = None
        self.model = None
        self.device = "cpu"
        self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Dependency torch dan transformers diperlukan untuk abstractive summarizer."
            ) from exc

        logging.info("Memuat model abstractive: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        logging.info("Model abstractive berjalan di: %s", self.device)

    def summarize(self, text: str) -> dict:
        """Summarize a single text and return metadata."""
        source = text.strip() if isinstance(text, str) else ""
        if not source:
            return self._result("", "empty_input", source, False, "")

        try:
            used_chunking = self._needs_chunking(source)
            if used_chunking:
                summary = self._summarize_long_text(source)
            else:
                summary = self._generate_batch([source])[0]
            return self._result(summary, "generated_ok", source, used_chunking, "")
        except Exception as exc:
            logging.exception("Abstractive inference gagal.")
            return self._result("", "error", source, False, str(exc))

    def summarize_batch(self, texts: list[str]) -> list[dict]:
        """Summarize multiple texts. Long texts are handled one by one."""
        results: list[dict] = []
        pending: list[tuple[int, str]] = []
        output_slots: dict[int, dict] = {}

        for index, text in enumerate(texts):
            source = text.strip() if isinstance(text, str) else ""
            if not source or self._needs_chunking(source):
                output_slots[index] = self.summarize(source)
            else:
                pending.append((index, source))

        for start in range(0, len(pending), self.config.batch_size):
            batch = pending[start : start + self.config.batch_size]
            batch_texts = [item[1] for item in batch]
            try:
                summaries = self._generate_batch(batch_texts)
                for (index, source), summary in zip(batch, summaries):
                    output_slots[index] = self._result(summary, "generated_ok", source, False, "")
            except Exception as exc:
                logging.exception("Batch abstractive inference gagal.")
                for index, source in batch:
                    output_slots[index] = self._result("", "error", source, False, str(exc))

        for index in range(len(texts)):
            results.append(output_slots[index])
        return results

    def _needs_chunking(self, text: str) -> bool:
        encoded = self.tokenizer(
            text,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return len(encoded["input_ids"]) > self.config.max_input_length

    def _summarize_long_text(self, text: str) -> str:
        chunks = self._chunk_by_sentence(text)
        partial_summaries = []
        for start in range(0, len(chunks), self.config.batch_size):
            partial_summaries.extend(self._generate_batch(chunks[start : start + self.config.batch_size]))
        combined = " ".join(partial_summaries).strip()
        if combined and self._needs_chunking(combined):
            return self._summarize_long_text(combined)
        if combined:
            return self._generate_batch([combined])[0]
        return ""

    def _chunk_by_sentence(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip()
            if current and self._needs_chunking(candidate):
                chunks.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks or [text]

    def _generate_batch(self, texts: list[str]) -> list[str]:
        import torch

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_input_length,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=self.config.num_beams,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                early_stopping=True,
                max_length=self.config.max_summary_length,
                min_length=self.config.min_summary_length,
            )
        return [self.tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]

    def _result(
        self,
        summary: str,
        status: str,
        source: str,
        used_chunking: bool,
        error: str,
    ) -> dict:
        # status="generated_ok" means the model call completed without raising
        # an exception. It is a technical/mechanical status only - it does NOT
        # mean the summary content is factually or religiously correct. mT5 is
        # a general-purpose model and has been observed to hallucinate details
        # (names, places, ages) not present in the source text. Extractive
        # summaries remain the safer default output; abstractive is an
        # experimental comparison that always needs human review.
        return {
            "summary": summary,
            "status": status,
            "model_name": self.model_name,
            "source_word_count": word_count(source),
            "summary_word_count": word_count(summary),
            "compression_ratio": compression_ratio(summary, source),
            "used_chunking": used_chunking,
            "error": error,
        }

