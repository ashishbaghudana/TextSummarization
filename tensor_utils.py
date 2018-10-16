import torch

from language import EOD_TOKEN
from seq2seq import device


def indices_from_text(language, text):
    return [language.word2index[word] for word in text.split()]


def tensor_from_text(language, text):
    indices = indices_from_text(language, text)
    indices.append(EOD_TOKEN)
    return torch.tensor(indices, dtype=torch.lang, device=device).view(-1, 1)


def tensors_from_pair(input_lang, target_lang, pair):
    input_tensor = tensor_from_text(input_lang, pair[0])
    target_tensor = tensor_from_text(target_lang, pair[1])
    return (input_tensor, target_tensor)
