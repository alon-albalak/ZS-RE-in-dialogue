import logging
import json
import random
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def tokenize(text, tokenizer):
    # The unused tokens are unique to each model
    D = ["madeupword0001", "madeupword0002"]
    text_tokens = []
    textraw = [text]
    # slight improvement with removing colons
    # textraw = [text.replace(":","")]
    for delimiter in D:
        ntextraw = []
        for i in range(len(textraw)):
            t = textraw[i].split(delimiter)
            for j in range(len(t)):
                ntextraw += [t[j]]
                if j != len(t) - 1:
                    ntextraw += [delimiter]
        textraw = ntextraw
    text = []
    for t in textraw:
        if t in ["[unused1]", "[unused2]"]:
            text += [t]
        else:
            tokens = tokenizer.tokenize(t)
            for tok in tokens:
                text += [tok]
    return text


class Sample(object):
    """A single training/test sample for token/sequence classification"""

    def __init__(self, guid=None, prompt=None, dialogue=None, head=None, tail=None, relation=None, label=None):
        self.guid = guid
        self.prompt = prompt
        self.dialogue = dialogue
        self.head = head
        self.tail = tail
        self.relation = relation
        self.label = label


def is_speaker(a):
    a = a.split()
    return len(a) == 2 and a[0] == "speaker" and a[1].isdigit()


def rename(d, x, y):
    # replace
    unused = ["[unused1]", "[unused2]"]
    a = []
    if is_speaker(x):
        a += [x]
    else:
        a += [None]
    if x != y and is_speaker(y):
        a += [y]
    else:
        a += [None]
    for i in range(len(a)):
        if a[i] is None:
            continue
        d = d.replace(a[i] + ":", unused[i] + " :")
        if x == a[i]:
            x = unused[i]
        if y == a[i]:
            y = unused[i]
    return d, x, y


def load_dialogRE_relation_extraction(relations, templates, data_path="data_v2/full_dataset_with_identifiers.json", rename_entities=True, include_negative_samples=True, lowercase=False, num_negative_samples=3):
    logger.info(f"Loading relations {relations} from {data_path}")
    data = json.load(open(data_path))

    num_pos, num_neg = 0, 0
    samples = []
    for datum in data:
        for entity_pair in datum[1]:
            # check that at least 1 relation in this entity pair is from the set of relations we care about
            found = any(
                [relation in relations for relation in entity_pair["r"]])
            if not found:
                continue

            if rename_entities:
                if lowercase:
                    dialogue, head, tail = rename("\n".join(datum[0]).lower(
                    ), entity_pair["x"].lower(), entity_pair["y"].lower())
                else:
                    dialogue, head, tail = rename(
                        "\n".join(datum[0]), entity_pair["x"], entity_pair["y"])
            else:
                if lowercase:
                    dialogue = "\n".join(datum[0]).lower()
                    head = entity_pair["x"].lower()
                    tail = entity_pair["y"].lower()
                else:
                    dialogue = "\n".join(datum[0])
                    head = entity_pair["x"]
                    tail = entity_pair["y"]
            dialogue = convert_to_unicode(dialogue)
            head = convert_to_unicode(head)
            tail = convert_to_unicode(tail)

            # collect positive samples
            for relation in entity_pair["r"]:
                if relation in relations:
                    prompts = templates.fill_in_template(relation, head, tail)
                    for p in prompts:
                        num_pos += 1
                        samples.append(
                            Sample(entity_pair["guid"], convert_to_unicode(p), dialogue, head, tail, relation, 1))

            if include_negative_samples:
                # collect negative samples
                for relation in relations:
                    if relation not in entity_pair["r"]:
                        prompts = templates.fill_in_template(
                            relation, head, tail)
                        prompts = random.sample(prompts, min(
                            len(prompts), num_negative_samples))
                        for p in prompts:
                            num_neg += 1
                            samples.append(
                                Sample(entity_pair["guid"], p, dialogue, head, tail, relation, 0))

    logging.info(f"Positive samples: {num_pos} ** Negative samples: {num_neg}")
    return samples


class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, label, sample):
        self.input_ids = input_ids,
        self.attention_mask = attention_mask,
        self.label = label,
        self.sample = sample


def convert_samples_to_features(samples, max_sequence_len, tokenizer, model_token_correction=2, logging=True):
    """
    model_token_correction is to account for model specific tokens:
            eg. 2 for RoBERTa - <s> Prompt </s> Dialogue
    """

    def _truncate_sequence(
        max_sequence_len, prompt_tokens, dialogue_tokens, model_token_correction=2
    ):
        """
        Truncate a sequence so that it will fit into the model
        model_token_correction is to account for model specific tokens:
                eg. 2 for RoBERTa - <s> Prompt </s> Dialogue
        """
        truncated = False
        sequence_len = len(dialogue_tokens) + len(prompt_tokens)
        adjusted_max_sequence_len = max_sequence_len - model_token_correction

        while sequence_len > adjusted_max_sequence_len:
            truncated = True
            if len(dialogue_tokens) > 0:
                dialogue_tokens.pop()
            elif len(prompt_tokens) > 0:
                prompt_tokens.pop()

            sequence_len = len(dialogue_tokens) + len(prompt_tokens)
        return truncated

    if logging:
        logger.info(f"Converting {len(samples)} samples to features")

    features = []
    num_truncated = 0

    for sample in samples:
        prompt_tokens = tokenize(sample.prompt, tokenizer)
        dialogue_tokens = tokenize(sample.dialogue, tokenizer)

        truncated = _truncate_sequence(
            max_sequence_len, prompt_tokens, dialogue_tokens, model_token_correction)
        if truncated:
            num_truncated += 1

        tokens = []
        tokens.append(tokenizer.cls_token)
        tokens.extend(prompt_tokens)
        tokens.append(tokenizer.sep_token)
        tokens.extend(dialogue_tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1]*len(input_ids)

        while len(input_ids) < max_sequence_len:
            input_ids.append(tokenizer.pad_token_id)
            attention_mask.append(0)

        assert len(input_ids) == max_sequence_len
        assert len(attention_mask) == max_sequence_len

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                label=sample.label,
                sample={
                    "guid": sample.guid,
                    "prompt": sample.prompt,
                    "head": sample.head,
                    "tail": sample.tail,
                    "relation": sample.relation,
                    "label": sample.label
                }
            )
        )

    if logging:
        logger.info(f"Truncated {num_truncated} samples")
    return features


class TensorListDataset(Dataset):
    """Dataset wrapping tensors, tensor dicts, and tensor lists

    *data (Tensor or dict or list of Tensors): tensors that all have the same size in the first dimension
    """

    def __init__(self, *data):
        if isinstance(data[0], dict):
            size = list(data[0].values())[0].size(0)
        elif isinstance(data[0], list):
            if isinstance(data[0][0], str):
                size = len(data[0])
            else:
                size = data[0][0].size(0)
        else:
            size = data[0].size(0)
        for element in data:
            if isinstance(element, dict):
                if isinstance(list(element.values())[0], list):
                    assert all(size == len(l)
                               for name, l in element.items())  # dict of lists
                else:
                    assert all(size == tensor.size(0)
                               for name, tensor in element.items())  # dict of tensors

            elif isinstance(element, list):
                if element and isinstance(element[0], str):
                    continue
                if element and isinstance(element[0], dict):
                    continue
                if element and isinstance(element[0], list):
                    continue
                assert all(size == tensor.size(0)
                           for tensor in element)  # list of tensors
            else:
                assert size == element.size(0)  # tensor
        self.size = size
        self.data = data

    def __getitem__(self, index):
        result = []
        for element in self.data:
            if isinstance(element, dict):
                result.append({k: v[index] for k, v in element.items()})
            elif isinstance(element, list):
                if isinstance(element[index], str):
                    result.append(element[index])
                elif isinstance(element[index], list):
                    result.append(element[index])
                elif isinstance(element[index], dict):
                    result.append(element[index])
                else:
                    result.append(v[index] for v in element)
            else:
                result.append(element[index])
        return tuple(result)

    def __len__(self):
        return self.size
