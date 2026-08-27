"""Markdown-aware chunking that preserves section meaning and fenced code blocks."""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from typing import Iterable

from rag_ingestion.models import Chunk, SourceDocument

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True, slots=True)
class MarkdownBlock:
    """An atomic unit that must stay within its original documentation section."""

    content: str
    header_path: tuple[str, ...]
    is_code: bool = False


def estimate_tokens(text: str) -> int:
    """A fast, dependency-free approximation adequate for bounded retrieval chunks."""

    return len(_TOKEN.findall(text))


class MarkdownChunker:
    """Build chunks without splitting fenced code or merging different headings."""

    def __init__(self, max_tokens: int = 350, overlap_tokens: int = 40) -> None:
        if max_tokens < 32:
            raise ValueError("max_tokens must be at least 32")
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def parse(self, markdown: str) -> list[MarkdownBlock]:
        """Parse prose, headings, and fenced code into non-destructive blocks."""

        blocks: list[MarkdownBlock] = []
        header_stack: list[str] = []
        prose_lines: list[str] = []
        code_lines: list[str] = []
        in_fence = False
        fence_marker = ""

        def flush_prose() -> None:
            text = "\n".join(prose_lines).strip()
            prose_lines.clear()
            if text:
                blocks.append(MarkdownBlock(text, tuple(header_stack)))

        for raw_line in markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            fence_match = _FENCE.match(raw_line)
            if in_fence:
                code_lines.append(raw_line)
                if fence_match and raw_line.lstrip().startswith(fence_marker[0]):
                    if len(fence_match.group(1)) >= len(fence_marker):
                        blocks.append(MarkdownBlock("\n".join(code_lines), tuple(header_stack), True))
                        code_lines.clear()
                        in_fence = False
                        fence_marker = ""
                continue

            if fence_match:
                flush_prose()
                in_fence = True
                fence_marker = fence_match.group(1)
                code_lines = [raw_line]
                continue

            heading_match = _HEADING.match(raw_line)
            if heading_match:
                flush_prose()
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                header_stack[level - 1 :] = [title]
                continue

            if not raw_line.strip():
                flush_prose()
                continue
            prose_lines.append(raw_line)

        if in_fence:
            # A malformed/unfinished fence is still protected as a code block.
            blocks.append(MarkdownBlock("\n".join(code_lines), tuple(header_stack), True))
        else:
            flush_prose()
        return blocks

    def chunk(self, document: SourceDocument) -> list[Chunk]:
        """Chunk one document deterministically, retaining header breadcrumbs as context."""

        chunks: list[Chunk] = []
        pending: list[str] = []
        pending_tokens = 0
        pending_header: tuple[str, ...] | None = None

        def emit() -> None:
            nonlocal pending, pending_tokens, pending_header
            if not pending:
                return
            body = "\n\n".join(pending).strip()
            context = self._context(pending_header or ())
            content = f"{context}\n\n{body}" if context else body
            ordinal = len(chunks)
            stable_key = f"{document.source_url}:{ordinal}:{hashlib.sha256(content.encode()).hexdigest()}"
            chunks.append(
                Chunk(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key)),
                    ordinal=ordinal,
                    content=content,
                    header_path=pending_header or (),
                    token_count=estimate_tokens(content),
                )
            )
            pending = []
            pending_tokens = 0
            pending_header = None

        for block in self.parse(document.content):
            # Chunks never cross a heading boundary, even when room remains.
            if pending and pending_header != block.header_path:
                emit()

            context_tokens = estimate_tokens(self._context(block.header_path))
            budget = max(1, self.max_tokens - context_tokens)
            pieces = [block.content] if block.is_code else self._split_prose(block.content, budget)

            for piece in pieces:
                piece_tokens = estimate_tokens(piece)
                if pending and pending_tokens + piece_tokens > budget:
                    emit()
                if not pending:
                    pending_header = block.header_path
                pending.append(piece)
                pending_tokens += piece_tokens

                # An oversized code block remains whole but does not absorb prose.
                if block.is_code and piece_tokens > budget:
                    emit()
            # Preserve a small prose overlap only within the same section.
            if pending and self.overlap_tokens and not block.is_code:
                # Overlap is applied by _split_prose only when an individual prose block needs splitting.
                pass
        emit()
        return chunks

    @staticmethod
    def _context(header_path: tuple[str, ...]) -> str:
        return f"Section: {' > '.join(header_path)}" if header_path else ""

    def _split_prose(self, text: str, budget: int) -> list[str]:
        """Prefer sentence boundaries; only use word boundaries when a sentence is too large."""

        if estimate_tokens(text) <= budget:
            return [text]
        sentences = [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]
        if not sentences:
            sentences = [text]
        pieces: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_parts = self._split_words(sentence, budget)
            for part in sentence_parts:
                part_tokens = estimate_tokens(part)
                if current and current_tokens + part_tokens > budget:
                    pieces.append(" ".join(current))
                    current = self._tail_for_overlap(current)
                    current_tokens = estimate_tokens(" ".join(current))
                    # A full-size sentence cannot also carry an overlap.  Drop the
                    # overlap rather than producing an over-budget prose chunk.
                    if current and current_tokens + part_tokens > budget:
                        current = []
                        current_tokens = 0
                current.append(part)
                current_tokens += part_tokens
        if current:
            pieces.append(" ".join(current))
        return pieces

    @staticmethod
    def _split_words(text: str, budget: int) -> list[str]:
        if estimate_tokens(text) <= budget:
            return [text]
        words = text.split()
        parts: list[str] = []
        current: list[str] = []
        for word in words:
            if current and estimate_tokens(" ".join([*current, word])) > budget:
                parts.append(" ".join(current))
                current = []
            current.append(word)
        if current:
            parts.append(" ".join(current))
        return parts

    def _tail_for_overlap(self, parts: Iterable[str]) -> list[str]:
        """Return a word-aligned tail for adjacent chunks in one large prose block."""

        words = " ".join(parts).split()
        if not words or self.overlap_tokens == 0:
            return []
        tail: list[str] = []
        for word in reversed(words):
            candidate = [word, *tail]
            if estimate_tokens(" ".join(candidate)) > self.overlap_tokens:
                break
            tail = candidate
        return [" ".join(tail)] if tail else []
