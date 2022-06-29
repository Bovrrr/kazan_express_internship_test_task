from sklearn.base import TransformerMixin


import numpy as np
import pandas as pd


import torch


class ruBERTEmbeddings(TransformerMixin):
    def __init__(self, col_idx: int, model_name: str):
        from transformers import AutoTokenizer, AutoModel

        # Import generic wrappers

        self.col_idx = col_idx
        # вот тут надо исправить как-нибудь
        # можно костыльным способом - просто посмотреть на размерность
        # ембеддинга какого-нибудь универсального символа(например, точки)
        self.emb_size = 312  # from docs
        self.model_name = model_name
        self.train_vocab = set()

        # "cointegrated/rubert-tiny" - сейчас использую это

        # Define the model repo
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        if torch.cuda.is_available():
            self.model.cuda()

    def embed_bert_cls(self, text: str, model, tokenizer):
        t = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = model(**{k: v.to(model.device) for k, v in t.items()})
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings[0].cpu().numpy()

    def fit(self, X):
        if isinstance(X, pd.core.frame.DataFrame):
            for line in X.iloc[:, self.col_idx]:
                self.train_vocab = self.train_vocab.union(*line.split())
        if isinstance(X, np.ndarray):
            for line in X[:, self.col_idx]:
                self.train_vocab = self.train_vocab.union(*line.split())
        return self

    def transform(self, X):
        new_X = np.zeros((X.shape[0], self.emb_size))

        if isinstance(X, pd.core.frame.DataFrame):
            for i, line in enumerate(X.iloc[:, self.col_idx]):
                modif_line = " ".join(
                    list(set(line.split()).intersection(self.train_vocab))
                )
                new_X[i] = self.embed_bert_cls(modif_line, self.model, self.tokenizer)
            return pd.concat(
                [
                    X.drop(X.iloc[:, self.col_idx].name, axis=1),
                    pd.DataFrame(new_X, X.index),
                ],
                axis=1,
            )
        elif isinstance(X, np.ndarray):
            for i, line in enumerate(X[:, self.col_idx]):
                modif_line = " ".join(
                    list(set(line.split()).intersection(self.train_vocab))
                )
                new_X[i] = self.embed_bert_cls(modif_line, self.model, self.tokenizer)
            return np.vstack(
                [
                    X[:, list(set(np.arange(X.shape[1])).difference([self.col_idx]))],
                    new_X,
                ]
            )

        else:
            raise NotImplementedError

    def fit_transform(self, X):
        self.fit(X)
        self.transform(X)
