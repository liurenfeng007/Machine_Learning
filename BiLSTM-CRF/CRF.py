import torch
import torch.nn as nn
from torch.autograd import Variable

START_TAG = "START"
STOP_TAG = "STOP"

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()

def prepare_sequence(seq, word_to_idx):
    idxs = [word_to_idx[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return Variable(tensor)

def log_sum_exp(vec):
    # vec 2D: 1 * tagset_size
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size,  embedding_dim, hidden_dim, batch_size, tag_dict, dropout, pre_word_embeds=None):
        super(BiLSTM_CRF, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.tag_dict = tag_dict
        self.tag_size = len(tag_dict)

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        if pre_word_embeds is not None:
            self.pre_word_embeds = True
            self.word_embeds.weight = nn.Parameter(torch.FloatTensor(pre_word_embeds))
        else:
            self.pre_word_embeds = False
        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)
        self.transitions = nn.Parameter(torch.randn(self.tag_size, self.tag_size))

        self.transitions.data[tag_dict[START_TAG], :] = -10000
        self.transitions.data[:, tag_dict[STOP_TAG]] = -10000
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, self.batch_size, self.hidden_dim // 2),
                torch.randn(2, self.batch_size, self.hidden_dim // 2))

    def _get_lstm_features(self, sentence):
        sentence = Variable(sentence)
        self.hidden = self.init_hidden()
        length = sentence.shape[1]
        # print(sentence.size())
        embeds = self.word_embeds(sentence).view(length, -1, self.embedding_dim)
        # print(embeds.size())
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # print(lstm_out.size())
        lstm_out = lstm_out.view(self.batch_size, -1, self.hidden_dim)
        lstm_out = self.dropout(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)
        # print(lstm_feats.size())
        return lstm_feats

    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_dict[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_dict[STOP_TAG], tags[-1]]
        return score

    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tag_size), -10000.)
        init_alphas[0][self.tag_dict[START_TAG]] = 0.

        forward_var = init_alphas

        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tag_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tag_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_val = forward_var + self.transitions[self.tag_dict[STOP_TAG]]
        alpha = log_sum_exp(terminal_val)
        return alpha

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tag_size), -10000.)
        init_vvars[0][self.tag_dict[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.tag_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_dict[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        start = best_path.pop()
        # print(start)
        # print(self.tag_dict[START_TAG])
        # assert start == self.tag_dict[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentences, tags, length):
        feats = self._get_lstm_features(sentences)
        forward_score = torch.zeros(1)
        real_path_score = torch.zeros(1)
        for feat, tag, leng in zip(feats, tags, length):
            feat = feat[:leng]
            tag = tag[:leng]
            real_path_score += self._score_sentence(feat, tag)
            # print(real_path_score)
            forward_score += self._forward_alg(feat)
            # print(forward_score)
        return forward_score - real_path_score

    def forward(self, sentences, lengths):
        sentences = torch.LongTensor(sentences)
        # print(lengths)
        feats = self._get_lstm_features(sentences)

        scores = []
        paths = []
        for feat, leng in zip(feats, lengths):
            feat = feat[:leng]
            score, tag_seq = self._viterbi_decode(feat)
            # print(feat)
            # print(leng)
            scores.append(score)
            paths.append(tag_seq)

        return scores, paths

# if __name__ == '__main__':
