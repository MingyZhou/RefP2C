import re
from typing import List, Tuple


class MarkdownParser:
    def __init__(self, markdown_content: str):
        self._content = markdown_content
        self._raw_paragraphs = None
        self._clean_paragraphs = None
        self._sentences_by_paragraph = None

    @staticmethod
    def _split_sentences(paragraph: str) -> List[str]:
        protected = []

        def protect(match):
            protected.append(match.group(0))
            return f"__PROTECTED_{len(protected)-1}__"

        pattern = re.compile(r"(\$\$.*?\$\$|\$.*?\$|```.*?```|`.*?`)")
        text = pattern.sub(protect, paragraph)

        sentence_endings = re.compile(r"(?<!\w\\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s")
        sentences = sentence_endings.split(text)

        restored_sentences = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            for i, p in enumerate(protected):
                s = s.replace(f"__PROTECTED_{i}__", p)
            restored_sentences.append(s)

        return restored_sentences

    def _extract_paragraphs(self) -> List[str]:
        paragraphs = []
        current_para = []
        in_references_section = False

        if not self._content:
            return []

        for line in self._content.splitlines():
            stripped_line = line.strip()

            if stripped_line.startswith("\\section*{References}"):
                in_references_section = True
                if current_para:
                    paragraphs.append(" ".join(current_para).strip())
                current_para = []
                continue

            if in_references_section and stripped_line.startswith("\\section"):
                in_references_section = False

            if in_references_section:
                continue

            if stripped_line == "":
                if current_para:
                    paragraphs.append(" ".join(current_para).strip())
                    current_para = []
            elif stripped_line.startswith("\\section"):
                if current_para:
                    paragraphs.append(" ".join(current_para).strip())
                current_para = [stripped_line]
            else:
                current_para.append(stripped_line)

        if current_para and not in_references_section:
            paragraphs.append(" ".join(current_para).strip())

        return paragraphs

    def get_parsing_results(self) -> Tuple[List[str], List[str], List[List[str]]]:
        if self._raw_paragraphs is not None:
            return (
                self._raw_paragraphs,
                self._clean_paragraphs,
                self._sentences_by_paragraph,
            )

        if not self._content:
            return [], [], []

        raw_paragraphs = self._extract_paragraphs()

        clean_paragraphs = []
        sentences_by_paragraph = []

        for para in raw_paragraphs:
            if not para.startswith("\\section") and not para.startswith("\\subsection"):
                clean_paragraphs.append(para)

        for para in clean_paragraphs:
            sentences = self._split_sentences(para)
            sentences_by_paragraph.append(sentences)

        self._raw_paragraphs = raw_paragraphs
        self._clean_paragraphs = clean_paragraphs
        self._sentences_by_paragraph = sentences_by_paragraph

        return raw_paragraphs, clean_paragraphs, sentences_by_paragraph
