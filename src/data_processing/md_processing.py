import re


def split_markdown_sentences(paragraph):
    protected = []

    def protect(match):
        protected.append(match.group(0))
        return f"PROTECTED_{len(protected)-1}_"

    pattern = re.compile(r"(\$\$.*?\$\$|\$.*?\$|```.*?```)")
    text = pattern.sub(protect, paragraph)
    sentence_endings = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+")
    sentences = sentence_endings.split(text)
    result = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        for i, p in enumerate(protected):
            s = s.replace(f"PROTECTED_{i}_", p)
        result.append(s)

    return result


def extract_paragraphs_from_md(markdown_path: str):
    paragraphs = []
    current_para = []
    in_custom_block = False
    in_references_section = False

    with open(markdown_path, "r", encoding="utf-8") as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line.startswith("\\section*{References}"):
                in_references_section = True
                if current_para:
                    final_para = (
                        "\n".join(current_para).strip()
                        if in_custom_block
                        else " ".join(current_para).strip()
                    )
                    if final_para:
                        paragraphs.append(final_para)
                    current_para = []
                continue

            if in_references_section and stripped_line.startswith("\\section"):
                in_references_section = False
            if in_references_section:
                continue

            if in_custom_block:
                current_para.append(line.rstrip("\n"))
            else:
                if stripped_line == "":
                    if current_para:
                        paragraphs.append(" ".join(current_para).strip())
                        current_para = []
                elif stripped_line.startswith("\\section"):
                    if current_para:
                        paragraphs.append(" ".join(current_para).strip())
                        current_para = []
                    current_para.append(stripped_line)
                else:
                    current_para.append(stripped_line)

    if current_para:
        if not in_references_section:
            final_para = (
                "\n".join(current_para).strip()
                if in_custom_block
                else " ".join(current_para).strip()
            )
            paragraphs.append(final_para)

    return paragraphs


def extract_sentence_from_md(path):
    sentence = []
    paragraphs = extract_paragraphs_from_md(path)
    clean_paragraphs = []
    for i in paragraphs:
        if not i.startswith("\\section") and not i.startswith("\\subsection"):
            clean_paragraphs.append(i)

    for i in clean_paragraphs:
        all_sentences = split_markdown_sentences(i)
        sentence.append(all_sentences)
    return paragraphs, clean_paragraphs, sentence


if __name__ == "__main__":
    paragraphs = extract_paragraphs_from_md("./APT/paper.md")
    count = 1
    for i in paragraphs:
        print("=" * 20, count)
        if ":::" in i:
            print(i)
        else:
            all_sentences = split_markdown_sentences(i)
            for j in range(len(all_sentences)):
                print(f"{j}:{all_sentences[j]}")
        count += 1
