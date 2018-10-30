import random
import torch

from language import SOD_TOKEN, EOD_TOKEN
from seq2seq import MAX_LENGTH, device
from tensor_utils import tensor_from_text
from logging_utils import get_logger


LOGGER = get_logger('seq2seq.evaluate')


def evaluate(input_lang,
             target_lang,
             encoder,
             decoder,
             text,
             max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensor_from_text(input_lang, text)
        input_length = input_tensor.size()[0]

        encoder_hidden = encoder.init_hidden()
        encoder_outputs = torch.zeros(
            max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOD_TOKEN]], device=device)
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOD_TOKEN:
                decoded_words.append('<EOD>')
                break
            else:
                decoded_words.append(target_lang.index2word[topi.item()])
            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluate_randomly_training(pairs, encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        LOGGER.info('>', pair[0])
        LOGGER.info('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        LOGGER.info('<', output_sentence)
