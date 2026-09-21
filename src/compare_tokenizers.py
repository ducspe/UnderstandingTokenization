from pathlib import Path

import tiktoken

from .basic_bpe import BasicBPETokenizer
from .regex_bpe import RegexBPETokenizer
from .gpt4_tokenizer_wrapper import GPT4TokenizerWrapper


def main():
    project_directory = Path.cwd()
    file_path = project_directory / "william_shakespeare_wikipedia_article.txt"
    file_text = file_path.read_text(encoding="utf-8")
    article_text = file_text.split("\n---\n\n", 1)[1]

    paragraphs = article_text.split("\n\n")
    split_at = int(0.8 * len(paragraphs))
    training_text = "\n\n".join(paragraphs[:split_at])
    test_text = "\n\n".join(paragraphs[split_at:])

    vocab_size = 512
    basic_bpe_tokenizer = BasicBPETokenizer()
    regex_bpe_tokenizer = RegexBPETokenizer()
    for tokenizer in [basic_bpe_tokenizer, regex_bpe_tokenizer]:
        tokenizer.train(training_text, vocab_size=vocab_size)

    gpt4_tokenizer_wrapper = GPT4TokenizerWrapper()

    for tokenizer in [basic_bpe_tokenizer, regex_bpe_tokenizer, gpt4_tokenizer_wrapper]:
        token_ids = tokenizer.encode(test_text)
        decoded_text = tokenizer.decode(token_ids)

        assert decoded_text == test_text
        print(f"{type(tokenizer).__name__}: {len(token_ids):,} tokens")

    special_text = (
        "<|endoftext|>To be, or not to be? 🌙\n"
        "<|fim_prefix|>A café <|fim_suffix|>under the stars."
        "<|fim_middle|>awaits <|endofprompt|>"
    )
    special_ids = gpt4_tokenizer_wrapper.encode(special_text, allowed_special="all")
    reference_tokenizer = tiktoken.get_encoding("cl100k_base")
    assert special_ids == reference_tokenizer.encode(special_text, allowed_special="all")
    assert gpt4_tokenizer_wrapper.decode(special_ids) == special_text
    print("GPT4TokenizerWrapper: special-token encoding and decoding checks passed.")

    output_directory = project_directory / "tokenizer_outputs"
    output_directory.mkdir(exist_ok=True)
    basic_bpe_tokenizer.save(str(output_directory / "basic_bpe"))
    regex_bpe_tokenizer.save(str(output_directory / "regex_bpe"))
    gpt4_tokenizer_wrapper.save_vocab(output_directory / "gpt4.vocab")
    print("Vocabulary files saved to:", output_directory)


if __name__ == "__main__":
    main()
