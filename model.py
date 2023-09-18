import flair
import torch
import torch.nn as nn
import torch.nn.functional as F
from flair.data import Sentence, Dictionary
from flair.embeddings import TransformerDocumentEmbeddings, DocumentRNNEmbeddings, TokenEmbeddings
from flair.models import TextClassifier as FlairTextClassifier
from typing import List, Union, Dict, Optional, cast, Any


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=2):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attention_weights = nn.Parameter(torch.randn(num_heads, hidden_dim), requires_grad=True)

    def forward(self, rnn_output):
        attention_scores = torch.matmul(rnn_output, self.attention_weights.t())
        attention_distribution = F.softmax(attention_scores, dim=2).transpose(1, 2)
        weighted_rnn_output = torch.bmm(attention_distribution, rnn_output)
        return weighted_rnn_output


class DocumentRNNEmbeddingsWithAttention(DocumentRNNEmbeddings):
    def __init__(
            self,
            embeddings,
            hidden_size=128,
            rnn_type="GRU",
            bidirectional=True,
            dropout=0.5,
            word_dropout=0.0,
            locked_dropout=0.0,
            reproject_words=True,
            reproject_words_dimension=None,
            rnn_layers=1,
    ):
        super().__init__(
            embeddings=embeddings,
            hidden_size=hidden_size,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            dropout=dropout,
            word_dropout=word_dropout,
            locked_dropout=locked_dropout,
            reproject_words=reproject_words,
            reproject_words_dimension=reproject_words_dimension,
            rnn_layers=rnn_layers,
        )

    def forward(self, sentences):
        self.rnn_output = super().forward(sentences)

        # 将输入切分为原文和上下文
        sentences_splits = []
        for sentence in sentences:
            text = sentence.to_original_text()
            if "[EOS]" in text:
                text_parts = text.split("[EOS]")
                sentences_splits.append((text_parts[0], text_parts[1]))
            else:
                sentences_splits.append((text, None))

        # 分别计算原文和上下文的注意力加权表示
        weighted_rnn_outputs = []
        for text_part, context_part in sentences_splits:
            # 获取原文和上下文的句子表示
            text_sentence = Sentence(text_part)
            context_sentence = Sentence(context_part) if context_part else None

            # 分别计算原文和上下文的注意力加权表示
            text_weighted_rnn_output = self._compute_weighted_rnn_output(text_sentence)
            if context_sentence:
                context_weighted_rnn_output = self._compute_weighted_rnn_output(context_sentence)
            else:
                context_weighted_rnn_output = torch.zeros_like(text_weighted_rnn_output)

            # 将原文和上下文的表示拼接在一起
            weighted_rnn_output = torch.cat([text_weighted_rnn_output, context_weighted_rnn_output], dim=1)
            weighted_rnn_outputs.append(weighted_rnn_output)

        embeddings = torch.stack(weighted_rnn_outputs)
        return embeddings

    def _compute_weighted_rnn_output(self, sentence):
        sentence_embedding = self.embed(sentence)
        rnn_output = self.rnn(sentence_embedding.unsqueeze(1))[0]
        attention_weights = self.multi_head_attention(rnn_output)
        weighted_rnn_output = torch.sum(attention_weights, dim=1)
        return weighted_rnn_output

    def _get_state_dict(self):
        state = super()._get_state_dict()
        state["attention_weights"] = self.attention_weights
        return state

    def _load_state_dict(self, state):
        self.attention_weights = state["attention_weights"]
        super()._load_state_dict(state)


class TextClassifier(FlairTextClassifier):
    def __init__(
            self,
            document_embeddings: TokenEmbeddings,
            label_dictionary: Dictionary,
            multi_label: bool,
            multi_label_threshold: float = 0.5,
            state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__(document_embeddings, label_dictionary, multi_label, multi_label_threshold)

        self.decoder = nn.Linear(
            2 * self.document_embeddings.embedding_length, len(self.label_dictionary)
        )

        self.loss_function = (
            nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()
        )

        if state_dict is not None:
            self.load_state_dict(state_dict)

    @classmethod
    def _init_model_with_state_dict(cls, state_dict: Dict[str, Any], **kwargs) -> "Model":
        # Initialize the model
        model = cls(**kwargs)
        model.load_state_dict(state_dict)
        return model