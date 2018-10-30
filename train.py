import time
import os
import torch
import random
import torch.nn as nn
from torch import optim
from argparse import ArgumentParser
from pathlib import Path

from data_loader import DataLoader
from language import SOD_TOKEN, EOD_TOKEN
from seq2seq import device, MAX_LENGTH, EncoderRNN, AttentionDecoderRNN
from utils import time_since, time_string
from tensor_utils import tensors_from_pair
from logging_utils import get_logger
from evaluate import evaluate_randomly_training

teacher_forcing_ratio = 0.5

LOGGER = get_logger('seq2seq.train')


def train_tensor(input_tensor,
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


def train_epoch(training_pairs,
                encoder,
                decoder,
                print_every=1000,
                plot_every=100,
                learning_rate=0.001):

    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0
    n_steps = len(training_pairs)

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for step in range(1, n_steps + 1):
        training_pair = training_pairs[step - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train_tensor(input_tensor, target_tensor, encoder, decoder,
                            encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if step % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            LOGGER.info('Iterations complete = %s/%s' % (step, n_steps))
            LOGGER.info('Loss = %s' % print_loss_avg)
            LOGGER.debug(
                '%s (%d %d%%) %.4f' % (time_since(start, step / n_steps), step,
                                       step / n_steps * 100, print_loss_avg))

        if step % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


def save_models(encoder, decoder, learning_rate, epoch, directory):
    current_time = time.time()
    encoder_dir = os.path.join(directory, 'model_%s' % int(current_time),
                               'encoder')
    decoder_dir = os.path.join(directory, 'model_%s' % int(current_time),
                               'decoder')
    path_encoder = Path(encoder_dir)
    path_decoder = Path(decoder_dir)
    path_encoder.mkdir(parents=True, exist_ok=True)
    path_decoder.mkdir(parents=True, exist_ok=True)

    torch.save(
        encoder.state_dict(),
        os.path.join(encoder_dir, 'encoder_%s_%s.pt' % (epoch, learning_rate)))

    torch.save(
        decoder.state_dict(),
        os.path.join(decoder_dir, 'decoder_%s_%s.pt' % (epoch, learning_rate)))


def train(lang_1,
          lang_2,
          pairs,
          encoder,
          decoder,
          output_dir,
          n_epochs=500000,
          learning_rate=0.001,
          print_every=1000,
          save_every=5000):

    LOGGER.info('Starting training process...')

    save_every_epoch_start = time.time()

    training_pairs = [
        tensors_from_pair(lang_1, lang_2, pair) for pair in pairs
    ]

    for epoch in range(1, n_epochs + 1):

        start = time.time()

        LOGGER.debug('Start training epoch %i at %s' % (epoch, time_string()))

        # Train the particular step
        train_epoch(
            training_pairs,
            encoder,
            decoder,
            print_every=print_every,
            learning_rate=learning_rate)

        LOGGER.debug(
            'Finished training epoch %i at %s' % (epoch, time_string()))
        LOGGER.debug('Time taken for epoch %i = %s' %
                     (epoch, time_since(start, epoch / n_epochs)))

        LOGGER.info('Evaluating on training set randomly...')
        evaluate_randomly_training(lang_1, lang_2, pairs, encoder, decoder)

        if epoch % save_every == 0:
            LOGGER.info('Saving model at epoch %i...' % epoch)
            LOGGER.info('Time taken for %i epochs = %s' %
                        (save_every,
                         time_since(save_every_epoch_start, epoch / n_epochs)))
            save_models(encoder, decoder, learning_rate, epoch, output_dir)


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
        "-o", "--output_dir", help="Path to save the model", required=True)
    parser.add_argument(
        "--hidden_units", help="Number of hidden units", type=int, default=256)
    parser.add_argument(
        "--dropout",
        help="Dropout value in Attention Decoder",
        type=float,
        default=0.1)
    parser.add_argument(
        "--trim_dataset",
        help="Trim the dataset to a small number for testing purposes",
        required=False,
        type=int)
    parser.add_argument(
        "--print_every",
        help="Print every n steps",
        default=1000,
        type=int,
        required=False)
    parser.add_argument(
        "--save_every",
        help="Save model every n epochs",
        default=5,
        required=False,
        type=int)
    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="Learning rate",
        default=0.001,
        type=float)
    parser.add_argument(
        "-n",
        "--n_epochs",
        help="Number of epochs to train for",
        default=50,
        type=int)

    args = parser.parse_args()
    data = DataLoader(args.text_dir, args.summary_dir)
    full_text_lang, summary_text_lang, pairs = data.load(
        trim=args.trim_dataset)

    LOGGER.info('Creating models...')
    encoder = EncoderRNN(full_text_lang.n_words, args.hidden_units).to(device)
    attention_decoder = AttentionDecoderRNN(
        args.hidden_units, summary_text_lang.n_words, args.dropout).to(device)

    train(
        lang_1=full_text_lang,
        lang_2=summary_text_lang,
        pairs=pairs,
        encoder=encoder,
        decoder=attention_decoder,
        output_dir=args.output_dir,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        print_every=args.print_every,
        save_every=args.save_every)


if __name__ == '__main__':
    main()
