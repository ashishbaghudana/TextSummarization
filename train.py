import time
import torch
import random
import torch.nn as nn
from torch import optim
from argparse import ArgumentParser

from data_loader import DataLoader
from language import SOD_TOKEN, EOD_TOKEN
from seq2seq import device, MAX_LENGTH, EncoderRNN, AttentionDecoderRNN
from utils import time_since
from tensor_utils import tensors_from_pair

teacher_forcing_ratio = 0.5


def train(input_tensor,
          target_tensor,
          encoder,
          decoder,
          encoder_optimizer,
          decoder_optimizer,
          criterion,
          max_length=MAX_LENGTH):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(
        max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOD_TOKEN]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = (True if random.random() < teacher_forcing_ratio else
                           False)

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOD_TOKEN:
                break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train_iters(input_lang,
                target_lang,
                pairs,
                encoder,
                decoder,
                n_iters,
                print_every=1000,
                plot_every=100,
                learning_rate=0.01):

    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [
        tensors_from_pair(input_lang, target_lang, random.choice(pairs))
        for _ in range(n_iters)
    ]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder,
                     encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(
                '%s (%d %d%%) %.4f' % (time_since(start, iter / n_iters), iter,
                                       iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


def main():
    parser = ArgumentParser("Train Seq2Seq Attention Model")
    parser.add_argument(
        "-f",
        "--text_dir",
        help="Path to all the full text documents",
        required=True)
    parser.add_argument(
        "-s",
        "--summary_dir",
        help="Path to all the summary documents",
        required=False)
    parser.add_argument(
        "--hidden_units", help="Number of hidden units", type=int, default=256)
    parser.add_argument(
        "--dropout",
        help="Dropout value in Attention Decoder",
        type=float,
        default=0.1)

    args = parser.parse_args()
    data = DataLoader(args.text_dir, args.summary_dir)
    full_text_lang, summary_text_lang, pairs = data.load()

    encoder = EncoderRNN(full_text_lang.n_words, args.hidden_units).to(device)
    attention_decoder = AttentionDecoderRNN(
        args.hidden_units, summary_text_lang.n_words, args.dropout)

    train_iters(
        full_text_lang,
        summary_text_lang,
        pairs,
        encoder,
        attention_decoder,
        len(pairs),
        print_every=5000)


if __name__ == '__main__':
    main()
