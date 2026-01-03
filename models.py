from util import load_names
import torch
import torch.nn.functional as F

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

class BigramLv2:

    def __init__(self):
        self.words = load_names()
        chars = sorted(list(set("".join(self.words))))
        self.chars = chars + ["."]
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        xs, ys = [], []
        for word in self.words:
            chs = ["."] + list(word) + ["."]
            for ch1, ch2 in zip(chs, chs[1:]):
                xs.append(self.stoi[ch1])
                ys.append(self.stoi[ch2])
        self.xs = torch.tensor(xs)
        self.ys = torch.tensor(ys)
        self.xenc = F.one_hot(self.xs, num_classes=self.vocab_size).float()
        self.W = None 

    def train(self, epochs=200, lr=5.0):
        self.W = torch.randn((self.vocab_size, self.vocab_size), requires_grad=True)
        for i in range(epochs):
            logits = self.xenc @ self.W
            counts = logits.exp()
            probs = counts / counts.sum(1, keepdim=True)
            loss = -probs[torch.arange(len(self.xs)), self.ys].log().mean()
            self.W.grad = None
            loss.backward()
            self.W.data += -lr * self.W.grad

    def sample(self, n=10, max_len=10):
        if self.W is None:
            raise RuntimeError("Model must be trained before sampling")
        names = []
        for _ in range(n):
            ix = self.stoi["."]
            out = []
            while True:
                x = F.one_hot(torch.tensor([ix]), num_classes=self.vocab_size).float()
                logits = x @ self.W
                probs = logits.softmax(dim=1)
                ix = torch.multinomial(probs, num_samples=1).item()
                ch = self.itos[ix]
                if ch == "." or len(out) >= max_len:
                    break
                out.append(ch)
            names.append("".join(out))
        return names

    def evaluate(self, words=None):
        if self.W is None:
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
                x = F.one_hot(torch.tensor([i]), num_classes=self.vocab_size).float()
                probs = (x @ self.W).softmax(dim=1)
                log_likelihood += torch.log(probs[0, j])
                n += 1
        return (-log_likelihood / n).item()
    