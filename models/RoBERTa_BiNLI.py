from transformers import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from torch.nn import Dropout, Linear, BCEWithLogitsLoss, Sigmoid, Tanh
from torch import round as tround


class RoBERTa_BiNLI(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dense = Linear(config.hidden_size, config.hidden_size)
        self.dropout = Dropout(config.dropout_rate)
        self.classifier = Linear(config.hidden_size, 1)
        self.loss_fct = BCEWithLogitsLoss(reduction="none")
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        self.init_weights()

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids,
                               attention_mask=attention_mask, return_dict=False)
        cls_tokens = outputs[0][:, 0, :]
        x = self.dropout(cls_tokens)
        x = self.dense(x)
        x = self.tanh(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits

    def calculate_loss(self, input_ids, attention_mask, labels):
        logits = self.forward(input_ids, attention_mask)
        per_sample_loss = self.loss_fct(logits, labels)
        # if weighting positive samples, add line here something like: weight = labels*pos_weight*per_sample_loss
        return per_sample_loss

    def predict(self, input_ids, attention_mask):
        logits = self.forward(input_ids, attention_mask)
        preds = tround(self.sigmoid(logits))
        return preds
