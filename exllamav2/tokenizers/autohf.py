from typing import List, Union
from exllamav2.tokenizers.base import ExLlamaV2TokenizerBase
from transformers import AutoTokenizer

class ExLlamaV2TokenizerAutoHF(ExLlamaV2TokenizerBase):

    space_char_: str = " "
    newline_char_: str = "\n"

    def __init__(self, pretrained_model_name_or_path: str) -> None:
        super().__init__()

        self.hf_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

        self.space_char_ = self.deduce_char_map(" ")  # "Ġ"
        self.newline_char_ = self.deduce_char_map("\n")  # "Ċ"

    def unk_id(self) -> int or None: return self.hf_tokenizer.unk_token_id
    def pad_id(self) -> int or None: return self.hf_tokenizer.pad_token_id
    def bos_id(self) -> int or None: return self.hf_tokenizer.bos_token_id
    def eos_id(self) -> int or None: return self.hf_tokenizer.eos_token_id
    def unk_token(self) -> str or None: return self.hf_tokenizer.unk_token
    def pad_token(self) -> str or None: return self.hf_tokenizer.pad_token
    def bos_token(self) -> str or None: return self.hf_tokenizer.bos_token
    def eos_token(self) -> str or None: return self.hf_tokenizer.eos_token

    def space_char(self): return self.space_char_
    def newline_char(self): return self.newline_char_

    def enumerate_tokens(self):
        items = list(self.hf_tokenizer.get_vocab().items())
        return ((v, k) for k, v in items)

    def vocab_size(self) -> int:
        return self.hf_tokenizer.vocab_size

    def id_to_piece(self, idx: int) -> str:
        if idx is None: return ""
        return self.hf_tokenizer.convert_ids_to_tokens(idx)

    def piece_to_id(self, text: str) -> int:
        return self.hf_tokenizer.convert_tokens_to_ids(text)

    def decode(self, ids: List[int]) -> str:
        text = self.hf_tokenizer.decode(ids)
        return text

    def encode(self, text: list or str) -> list:
        encoding = self.hf_tokenizer.encode(text, add_special_tokens = False)
        return encoding

