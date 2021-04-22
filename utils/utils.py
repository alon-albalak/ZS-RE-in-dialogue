import logging
import argparse
import random
import numpy as np
from utils.templates import NLI_template, QA_template
from models.RoBERTa_BiNLI import RoBERTa_BiNLI
from transformers import RobertaConfig, RobertaTokenizer
import torch

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "NLI": (RobertaConfig, RoBERTa_BiNLI, RobertaTokenizer, NLI_template, 512)
}

relations = ['per:positive_impression', 'per:negative_impression', 'per:acquaintance', 'per:alumni', 'per:boss', 'per:subordinate', 'per:client', 'per:dates', 'per:friends', 'per:girl/boyfriend', 'per:neighbor', 'per:roommate', 'per:children', 'per:other_family', 'per:parents', 'per:siblings', 'per:spouse', 'per:place_of_residence',
             'per:place_of_birth', 'per:visited_place', 'per:origin', 'per:employee_or_member_of', 'per:schools_attended', 'per:works', 'per:age', 'per:date_of_birth', 'per:major', 'per:place_of_work', 'per:title', 'per:alternate_names', 'per:pet', 'gpe:residents_of_place', 'gpe:visitors_of_place', 'gpe:births_in_place', 'org:employees_or_members', 'org:students']


def set_seed(seed):
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        logger.info(f"Setting seed to {seed}")


def batch_to_device(batch, device):
    batch_on_device = []
    for element in batch:
        if isinstance(element, dict):
            # if element does not contain tensors, do not move it to device
            if isinstance(list(element.values())[0], torch.Tensor):
                batch_on_device.append({k: v.to(device)
                                       for k, v in element.items()})
            else:
                batch_on_device.append(element)
        elif isinstance(element[0], str):
            batch_on_device.append(element)
        elif isinstance(element, list) and len(element) == 1 and isinstance(element[0], dict):
            batch_on_device.append(element[0])
        elif isinstance(element[0], dict):
            batch_on_device.append(element)
        else:
            batch_on_device.append(element.to(device))
    return tuple(batch_on_device)


def parse_args():
    parser = argparse.ArgumentParser()
    # general args
    parser.add_argument("--cpu_only", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # model args
    parser.add_argument("--model_name_or_path",
                        type=str, default="roberta-base")
    parser.add_argument("--base_model", type=str, default="roberta-base")
    parser.add_argument("--dropout_rate", type=float, default=0.1)

    # training args
    parser.add_argument("-e", "--num_epochs", type=int, default=5)
    parser.add_argument("--effective_batch_size", type=int, default=28)
    parser.add_argument("--gpu_batch_size", type=int, default=7)
    parser.add_argument("-lr", "--learning_rate", type=float, default=3e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--pos_sample_weight", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--warmup_proportion", type=float, default=0.)

    # data args
    parser.add_argument("--data_path", type=str,
                        default="data_v2/full_dataset_with_identifiers.json")
    parser.add_argument("--data_split", type=int, default=3)
    # misc. args
    parser.add_argument("--debugging", action="store_true")

    args = parser.parse_args()

    if not args.cpu_only:
        setattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
    else:
        setattr(args, "device", "cpu")

    return vars(args)
