from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch import tensor

import pandas as pd
from itertools import zip_longest
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # specify which GPU(s) to be used
os.chdir("~/GitHub/iecor/pyTorch Runs/eng_fra")

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open('data2/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(l.split('\t')[0]), normalizeString(l.split('\t')[1]), float(l.split('\t')[-1]), float(l.split('\t')[-2]), float(l.split('\t')[-3]), float(l.split('\t')[-4])] for l in lines]
    clade = []
    time = []
    firstnode = []
    proxnode = []
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
        for pair in pairs :
            clade.append(pair[1])
            time.append(pair[0])
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
        for pair in pairs :
            clade.append(pair[2])
            time.append(pair[3])
            firstnode.append(pair[4])
            proxnode.append(pair[5])
    return input_lang, output_lang, pairs, clade, time, firstnode, proxnode
    

with open("pie.txt", "r") as f:
  # Read all lines into a list of strings
    pie_data = [line.strip().split('\t') for line in f]
    pie_words = [normalizeString(pair[0]) for pair in pie_data]
    meanings = [pair[1] for pair in pie_data]

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs, input_clade, input_time, input_firstnode, input_proxnode = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    if reverse:
        for pair in pairs:
            input_lang.addSentence(pair[3])
            output_lang.addSentence(pair[2])
    else:
        for pair in pairs:
            input_lang.addSentence(pair[0])
            output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs, input_clade, input_time, input_firstnode, input_proxnode

#input_lang, output_lang, pairs, input_clade, input_time, input_firstnode, input_proxnode = prepareData('eng', 'fra', False)
#print(random.choice(pairs))

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1, num_copies=4):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
        # Additional input features
        self.input_clade_fc = nn.Linear(1, hidden_size)  # Assuming input_clade is a scalar value
        self.input_time_fc = nn.Linear(1, hidden_size)   # Assuming input_time is a scalar value
        self.input_firstnode_fc = nn.Linear(1, hidden_size)   # Assuming input_time is a scalar value
        self.input_proxnode_fc = nn.Linear(1, hidden_size)   # Assuming input_time is a scalar value
    def forward(self, input_ids, input_clade, input_time, input_firstnode, input_proxnode):
        embedded = self.dropout(self.embedding(input_ids))
        output, hidden = self.gru(embedded)
        # Process additional features
        input_clade_processed = self.input_clade_fc(input_clade[:, -2:])   # Slice input_clade along dimension 1
        input_time_processed = self.input_time_fc(input_time[:, -2:])
        input_firstnode_processed = self.input_firstnode_fc(input_firstnode[:, -2:])
        input_proxnode_processed = self.input_proxnode_fc(input_proxnode[:, -2:])
        input_clade_processed = input_clade_processed.unsqueeze(1)  # Expand dimensions to match with 'output' tensor
        input_time_processed = input_time_processed.unsqueeze(1)    # Expand dimensions to match with 'output' tensor
        input_firstnode_processed = input_firstnode_processed.unsqueeze(1)    # Expand dimensions to match with 'output' tensor
        input_proxnode_processed = input_proxnode_processed.unsqueeze(1)    # Expand dimensions to match with 'output' tensor
        # Concatenate the outputs of GRU and additional features
        output = torch.cat((output, input_clade_processed, input_time_processed, input_firstnode_processed, input_proxnode_processed), dim=1)   # Change this line from dim=1 to dim=2
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_copies=4):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)
            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop
    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden





class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)
    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []
        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)
            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)
        return decoder_outputs, decoder_hidden, attentions
    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))
        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)
        return output, hidden, attn_weights


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    clade_tensor = tensor(pair[2])
    time_tensor = tensor(pair[3])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, clade_tensor, time_tensor, target_tensor)

def get_dataloader(batch_size):
    input_lang, output_lang, pairs, clade, time, firstnode, proxnode = prepareData('eng', 'fra', False)  # Modify the return values here
    n = len(pairs)
    # Additional features: input_clade and input_time
    input_clade = np.array(clade).reshape(-1, 1)
    input_time = np.array(time).reshape(-1, 1)
    input_firstnode = np.array(firstnode).reshape(-1, 1)
    input_proxnode = np.array(proxnode).reshape(-1, 1)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    sub_pairs = [(x[0], x[1]) for x in pairs]
    for idx, (inp, tgt) in enumerate(sub_pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids
    # Stack the additional features with the existing tensors
    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
        torch.LongTensor(target_ids).to(device),
        torch.FloatTensor(input_clade).to(device),  # Add input_clade as a feature
        torch.FloatTensor(input_time).to(device),   # Add input_time as a feature
        torch.FloatTensor(input_firstnode).to(device),   # Add input_firstnode as a feature
        torch.FloatTensor(input_proxnode).to(device))   # Add input_proxnode as a feature
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader


def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):
    total_loss = 0
    for data in dataloader:
        input_ids, target_tensor, input_clade, input_time, input_firstnode, input_proxnode = data
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        encoder_outputs, encoder_hidden = encoder(input_ids, input_clade, input_time, input_firstnode, input_proxnode)  # Pass the new inputs to EncoderRNN
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
        loss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)), target_tensor.view(-1))
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001, print_every=100, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    best_loss = float('inf')  # Initialize best loss with infinity
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg))
        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
        # Save best loss and models checkpoint
        if loss < best_loss:
            best_loss = loss
            torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict(), 'loss': best_loss}, "~/GitHub/iecor/pyTorch Runs/second_best_model.pth")
    showPlot(plot_losses, "~/GitHub/iecor/pyTorch Runs/second_progress_1.png")

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points, filename):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig(filename)  # Save the figure to disk

def evaluate(encoder, decoder, sentence, input_lang, output_lang, input_clade, input_time, input_firstnode, input_proxnode):
    with torch.no_grad():
        input_clade = tensor(np.array(input_clade, dtype=np.float32).reshape(-1, 1)).to(device)
        input_time = tensor(np.array(input_time, dtype=np.float32).reshape(-1, 1)).to(device)
        input_firstnode = tensor(np.array(input_firstnode, dtype=np.float32).reshape(-1, 1)).to(device)
        input_proxnode = tensor(np.array(input_proxnode, dtype=np.float32).reshape(-1, 1)).to(device)
        input_tensor = tensorFromSentence(input_lang, sentence).to(device)
        encoder_outputs, encoder_hidden = encoder(input_tensor, input_clade, input_time, input_firstnode, input_proxnode)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)
        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()
        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('from ', pair[3])
        print('in ', pair[2])
        print('=', pair[1])
        input_clade = pair[2]
        input_time = pair[3]
        output_words, _ = evaluate(encoder, decoder, str(pair[0]), input_lang, output_lang, input_clade, input_time, input_firstnode, input_proxnode)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
        
def newLang(clade, time, firstnode, proxnode):
    new_language = []
    for i in pie_words:
        output_words, _ = evaluate(encoder, decoder, i, input_lang, output_lang, clade, time, firtnode, proxnode)
        output_sentence = ' '.join(output_words)
        output_parts = output_sentence.split('<EOS>')
        first_result = output_parts[0]
        result_parts = first_result.split('SOS')
        result = result_parts[0]
        new_language.append(result)
    return new_language

def newLangFrame(clade, time):
    new_language = newLang(clade, time)
    #pie_words = [sublist for sublist in pie_words]
    #meanings = [sublist for sublist in meanings]
    #new_language = [sublist for sublist in new_language]
    data = list(zip_longest(pie_words, meanings, new_language))
    df = pd.DataFrame(data)
    return df


df.to_csv("~/Downloads/newRomance.csv")
new_lex = newLangFrame(23, 8120)
new_lex.to_csv("~/Downloads/newRomance.csv")

hidden_size = 128
batch_size = 32

input_lang, output_lang, train_dataloader = get_dataloader(batch_size)
input_lang2, output_lang2, train_dataloader2 = get_dataloader(1)


encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

train(train_dataloader, encoder, decoder, 10000, print_every=1, plot_every=1)

torch.save(encoder.state_dict(), "~/GitHub/iecor/pyTorch Runs/encoder_1.pth")
torch.save(decoder.state_dict(), "~/GitHub/iecor/pyTorch Runs/decoder_1.pth")

checkpoint = torch.load("~/GitHub/iecor/pyTorch Runs/best_model.pth")
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])


input_lang, output_lang, pairs, clade, time = prepareData('eng', 'fra', False)

encoder.eval()
decoder.eval()
evaluateRandomly(encoder, decoder)

def showAttention(input_sentence, output_words, attentions):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.cpu().numpy(), cmap='bone')
    fig.colorbar(cax)
    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)
    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(encoder, decoder, input_sentence, input_lang, output_lang)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions[0, :len(output_words), :])
