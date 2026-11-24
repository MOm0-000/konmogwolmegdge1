"""DistilBERT classification with configurable CLS, mean, or max pooling.

The encoder, classifier initialization and checkpoint interfaces are reused
from Hugging Face Transformers 4.40.2.  Only the classification forward pass
is customized for this experiment.
"""
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import DistilBertConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.distilbert.modeling_distilbert import (
    Embeddings,
    MultiHeadSelfAttention,
    FFN,
    TransformerBlock,
    Transformer,
    DistilBertPreTrainedModel,
    DistilBertModel,
    DistilBertForMaskedLM,
    DistilBertForSequenceClassification as _BaseSequenceClassifier,
)

POOLING_METHODS = ("cls", "mean", "max")


def pool_hidden_states(
    hidden_state: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    pooling: str = "cls",
) -> torch.Tensor:
    """Convert [batch, sequence, hidden] to [batch, hidden]."""
    if pooling == "cls":
        return hidden_state[:, 0]
    if pooling not in POOLING_METHODS:
        raise ValueError(f"Unknown pooling method: {pooling}")

    if attention_mask is None:
        attention_mask = torch.ones(
            hidden_state.shape[:2], dtype=torch.bool, device=hidden_state.device
        )
    mask = attention_mask.unsqueeze(-1).to(
        device=hidden_state.device, dtype=torch.bool
    )

    if pooling == "mean":
        # Exclude padding tokens; keep CLS and SEP as valid tokens.
        masked_hidden = hidden_state.masked_fill(~mask, 0.0)
        count = mask.sum(dim=1).clamp(min=1).to(hidden_state.dtype)
        return masked_hidden.sum(dim=1) / count

    # For max pooling, padding cannot contribute to the maximum.
    masked_hidden = hidden_state.masked_fill(
        ~mask, torch.finfo(hidden_state.dtype).min
    )
    pooled = masked_hidden.max(dim=1).values
    # Also handle an entirely masked sequence safely.
    return torch.where(mask.any(dim=1), pooled, torch.zeros_like(pooled))


class DistilBertForSequenceClassification(_BaseSequenceClassifier):
    """Original DistilBERT classification head with configurable pooling."""

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]

        # The method is passed by train_cls.py and persisted in config.json.
        pooling = getattr(self.config, "pooling", "cls")
        pooled_output = pool_hidden_states(hidden_state, attention_mask, pooling)

        # Keep the original classification head unchanged.
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and labels.dtype in (torch.long, torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = CrossEntropyLoss()(
                    logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss = BCEWithLogitsLoss()(logits, labels)

        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )
