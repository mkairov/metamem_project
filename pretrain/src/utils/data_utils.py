import random
import string

from torch.utils.data import Dataset

from tokenizers import Tokenizer, Regex
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split

from transformers import PreTrainedTokenizerFast

# Define alphabets for character generation
ALPHABET = string.ascii_letters + string.digits
KV_ALPHABET = ALPHABET


def create_noisy_ar_tokenizer():
    # Create character tokenizer
    chars = ALPHABET + '!?:|'
    special = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2, '[UNK]': 3}
    vocab = {ch: i + len(special) for i, ch in enumerate(chars)}
    vocab.update(special)

    tokenizer = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Split(Regex(r'.'), behavior="isolated", invert=True)

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token='[PAD]',
        eos_token='[EOS]',
        bos_token='[BOS]',
        unk_token='[UNK]'
    )
