import logging
from utils import data_utils
import torch

logger = logging.getLogger(__name__)


def create_data(tokenizer, relations, templates, **kwargs):
    """
    calls functions which:
        load data and make examples
        convert examples to features
        save data if desired
    """

    samples = data_utils.load_dialogRE_relation_extraction(
        relations, templates, num_negative_samples=kwargs['num_negative_samples'])
    features = data_utils.convert_samples_to_features(
        samples, kwargs['max_sequence_len'], tokenizer)

    # if saving features, do so here
    return features


def get_data(tokenizer, relations, templates, **kwargs):
    features = create_data(tokenizer, relations, templates, **kwargs)
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long)
    labels = torch.tensor([f.label for f in features], dtype=torch.float)

    samples = [f.sample for f in features]

    all_features = [input_ids, attention_mask, labels, samples]
    dataset = data_utils.TensorListDataset(*all_features)

    return dataset
