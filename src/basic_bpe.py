from .tokenizer_base import Tokenizer, get_stats, merge


class BasicBPETokenizer(Tokenizer):

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        token_ids = list(text.encode("utf-8"))

        merges = {}
        vocab = {token_id: bytes([token_id]) for token_id in range(256)}
        for i in range(num_merges):
            stats = get_stats(token_ids)
            token_pair = max(stats, key=stats.get)
            new_token_id = 256 + i
            token_ids = merge(token_ids, token_pair, new_token_id)

            merges[token_pair] = new_token_id
            vocab[new_token_id] = vocab[token_pair[0]] + vocab[token_pair[1]]
            if verbose:
                print(
                    f"merge {i+1}/{num_merges}: {token_pair} -> {new_token_id} "
                    f"({vocab[new_token_id]}) had {stats[token_pair]} occurrences"
                )

        self.merges = merges
        self.vocab = vocab

    def decode(self, token_ids):
        text_bytes = b"".join(self.vocab[token_id] for token_id in token_ids)
        return text_bytes.decode("utf-8", errors="replace")

    def encode(self, text):
        token_ids = list(text.encode("utf-8"))
        while len(token_ids) >= 2:
            stats = get_stats(token_ids)
            token_pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if token_pair not in self.merges:
                break
            new_token_id = self.merges[token_pair]
            token_ids = merge(token_ids, token_pair, new_token_id)
        return token_ids
