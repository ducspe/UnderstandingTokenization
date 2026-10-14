import argparse
import re
from pathlib import Path

from .basic_bpe import BasicBPETokenizer
from .gpt4_tokenizer_wrapper import GPT4TokenizerWrapper
from .regex_bpe import RegexBPETokenizer


TABLE_HEADER = (
    "| Token | Basic_BPE position | Regex_BPE position | GPT-4 position | "
    "Regex_BPE − Basic_BPE | Regex_BPE − GPT-4 |\n"
    "|---|---:|---:|---:|---:|---:|"
)


def get_merge_positions(tokenizer, byte_mapping=None):
    positions = {}
    for position, token_id in enumerate(sorted(tokenizer.merges.values()), start=1):
        token_bytes = tokenizer.vocab[token_id]
        if byte_mapping is not None:
            token_bytes = bytes(byte_mapping[byte] for byte in token_bytes)
        positions.setdefault(token_bytes, position)
    return positions


def format_token(token_bytes):
    try:
        label = repr(token_bytes.decode("utf-8"))
    except UnicodeDecodeError:
        label = repr(token_bytes)
    backtick_runs = re.findall(r"`+", label)
    fence = "`" * (max(map(len, backtick_runs), default=0) + 1)
    label = label.replace("|", r"\|")
    return f"{fence}{label}{fence}"


def position_difference(first, second):
    if first is None or second is None:
        return "—"
    difference = first - second
    return f"{difference:+d}" if difference else "0"


def build_comparison_rows(basic_positions, regex_positions, gpt4_positions, all_tokens=False):
    tokens = set(basic_positions) | set(regex_positions)
    if all_tokens:
        tokens.update(gpt4_positions)
    tokens = sorted(tokens, key=lambda token: (
        basic_positions.get(token, float("inf")),
        regex_positions.get(token, float("inf")),
        gpt4_positions.get(token, float("inf")),
        token,
    ))
    rows = []
    for token in tokens:
        basic_position = basic_positions.get(token)
        regex_position = regex_positions.get(token)
        gpt4_position = gpt4_positions.get(token)
        cells = [
            format_token(token),
            str(basic_position) if basic_position is not None else "—",
            str(regex_position) if regex_position is not None else "—",
            str(gpt4_position) if gpt4_position is not None else "—",
            position_difference(regex_position, basic_position),
            position_difference(regex_position, gpt4_position),
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return tokens, rows


def main():
    parser = argparse.ArgumentParser(description="Compare where the same tokens appear in merge order.")
    parser.add_argument(
        "--all-tokens", action="store_true",
        help="Also include tokens found only in GPT-4 (about 100,000 rows in total).",
    )
    args = parser.parse_args()

    output_directory = Path.cwd() / "tokenizer_outputs"
    model_files = [output_directory / f"{name}.model" for name in ("basic_bpe", "regex_bpe")]
    missing_files = [path.name for path in model_files if not path.is_file()]
    if missing_files:
        parser.error(
            f"Missing {', '.join(missing_files)} in tokenizer_outputs/. "
            "Run python -m src.compare_tokenizers first from the repository root."
        )

    basic_tokenizer = BasicBPETokenizer()
    regex_tokenizer = RegexBPETokenizer()
    basic_tokenizer.load(str(model_files[0]))
    regex_tokenizer.load(str(model_files[1]))
    gpt4_tokenizer = GPT4TokenizerWrapper()

    basic_positions = get_merge_positions(basic_tokenizer)
    regex_positions = get_merge_positions(regex_tokenizer)
    gpt4_positions = get_merge_positions(gpt4_tokenizer, gpt4_tokenizer.inverse_byte_shuffle)
    tokens, rows = build_comparison_rows(
        basic_positions, regex_positions, gpt4_positions, all_tokens=args.all_tokens,
    )
    shared_tokens = set(basic_positions) & set(regex_positions)
    shared_by_all = shared_tokens & set(gpt4_positions)
    scope = (
        "All merged tokens from the three approaches are included."
        if args.all_tokens else
        "The table includes every token learned by Basic_BPE or Regex_BPE, "
        "with GPT-4 as a reference. Tokens found only in GPT-4 are omitted; "
        "run `python -m src.compare_merges --all-tokens` to include them."
    )
    report = "\n\n".join([
        "# Comparing token merge positions",
        "Do these tokenizers learn the same pieces of text, and where do those pieces "
        "appear in their merge order? Tokens match when their bytes match, even if "
        "different pairs produced them.",
        scope,
        f"Basic_BPE: **{len(basic_positions):,}** merged tokens. "
        f"Regex_BPE: **{len(regex_positions):,}**. GPT-4: **{len(gpt4_positions):,}**. "
        f"Basic_BPE and Regex_BPE share **{len(shared_tokens):,}** resulting tokens; "
        f"**{len(shared_by_all):,}** occur in all three. "
        f"This table contains **{len(rows):,}** tokens.",
        "## How to read the table",
        "- Position **1** is the first merge after the 256 base byte tokens. "
        "Base byte tokens and special tokens are excluded.\n"
        "- A positive difference means the token appears later in Regex_BPE; "
        "a negative difference means it appears earlier. Zero means the positions match.\n"
        "- `—` means the token is absent from that vocabulary, or a difference cannot be calculated.\n"
        "- Quotes make leading and trailing spaces visible. Escapes such as `\\n` "
        "represent control characters. Incomplete UTF-8 tokens use byte notation, "
        "such as `b'\\xe2\\x82'`, so distinct bytes remain distinguishable.\n"
        "- Rows follow Basic_BPE's merge order, then Regex_BPE-only tokens, "
        "then GPT-4-only tokens when included. If multiple merges produce the same "
        "bytes, the earliest position is shown.",
        "GPT-4 positions describe the order of its pretrained merge rules. "
        "These differences measure order, not elapsed time or tokenizer quality. "
        "GPT-4 has a different training history and a much larger vocabulary.",
        TABLE_HEADER + "\n" + "\n".join(rows),
    ]) + "\n"
    report_path = output_directory / "merge_comparison.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"Basic_BPE and Regex_BPE share {len(shared_tokens):,} resulting tokens.")
    preview_rows = [row for token, row in zip(tokens, rows) if token in shared_tokens]
    if preview_rows:
        print("Preview: up to 10 tokens shared by Basic_BPE and Regex_BPE, in Basic_BPE order.")
    else:
        preview_rows = rows
        print("Preview: up to 10 tokens from the comparison.")
    print(TABLE_HEADER)
    print("\n".join(preview_rows[:10]))
    print(f"\nFull table ({len(rows):,} tokens) saved to: {report_path}")


if __name__ == "__main__":
    main()
