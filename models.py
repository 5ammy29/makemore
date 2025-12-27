from util import load_names
import torch

class BigramLv1:

    def __init__(self):
        self.words = load_names()
        chars = sorted(list(set("".join(self.words))))
        self.chars = chars + ["."]
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.N = None  
        self.P = None 

    def train(self, smoothing=1):
        N = torch.zeros((self.vocab_size, self.vocab_size), dtype=torch.int32)
        for word in self.words:
            chs = ["."] + list(word) + ["."]
            for ch1, ch2 in zip(chs, chs[1:]):
                i = self.stoi[ch1]
                j = self.stoi[ch2]
                N[i, j] += 1
        P = (N + smoothing).float()
        P = P / P.sum(dim=1, keepdim=True)
        self.N = N
        self.P = P

    def sample(self, n=10, max_len=10):
        if self.P is None:
            raise RuntimeError("Model must be trained before sampling")
        names = []
        for i in range(n):
            ix = self.stoi["."]
            out = []
            while True:
                p = self.P[ix]
                ix = torch.multinomial(p, num_samples=1).item()
                ch = self.itos[ix]
                if ch == "." or len(out) >= max_len:
                    break
                out.append(ch)
            names.append("".join(out))
        return names
    
    def evaluate(self, words=None):
        if self.P is None:
            raise RuntimeError("Model must be trained before evaluation")
        if words is None:
            words = self.words
        log_likelihood = 0.0
        n = 0
        for w in words:
            chs = ["."] + list(w) + ["."]
            for ch1, ch2 in zip(chs, chs[1:]):
                i = self.stoi[ch1]
                j = self.stoi[ch2]
                prob = self.P[i, j]
                log_likelihood += torch.log(prob)
                n += 1
        avg_nll = (-log_likelihood / n).item()
        return avg_nll
