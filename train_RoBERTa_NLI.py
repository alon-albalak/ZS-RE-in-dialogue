import logging
import os
import sys
import json
from tqdm import tqdm
from utils import utils
from create_data import get_data
import torch
from transformers import get_linear_schedule_with_warmup

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def main(**kwargs):
    if kwargs["seed"] != -1:
        utils.set_seed(kwargs["seed"])

    kwargs['num_labels'] = 1
    config_class, model_class, tokenizer_class, templates_class, max_sequence_len = utils.MODEL_CLASSES[
        "NLI"]
    kwargs['max_sequence_len'] = max_sequence_len
    config = config_class.from_pretrained(kwargs['model_name_or_path'])
    config.update(kwargs)
    tokenizer = tokenizer_class.from_pretrained(kwargs['model_name_or_path'])
    templates = templates_class()

    if kwargs['debugging']:
        train_relations = [
            'per:positive_impression', 'per:employee_or_member_of']  # , 'per:place_of_birth', 'per:visited_place']
        dev_relations = ['per:acquaintance', 'per:alumni']
    else:
        data_splits = json.load(open("data_v2/data_splits.json"))
        train_relations = data_splits[kwargs['data_split']]["train"][0]
        dev_relations = data_splits[kwargs['data_split']]["dev"][0]

    train_dataset = get_data(tokenizer, train_relations, templates, **kwargs)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=kwargs['gpu_batch_size'], shuffle=True)

    dev_dataset = get_data(tokenizer, dev_relations, templates, **kwargs)
    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=kwargs['gpu_batch_size'], shuffle=False)

    # load model
    model = model_class.from_pretrained(
        kwargs['model_name_or_path'], config=config)
    model.to(kwargs['device'])

    # optimization vars
    gradient_accumulation_steps = kwargs["effective_batch_size"] / \
        kwargs["gpu_batch_size"]
    total_optimization_steps = kwargs["num_epochs"] * \
        (len(train_dataloader) // gradient_accumulation_steps)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=kwargs["learning_rate"])

    if kwargs['warmup_proportion'] > 0:
        num_warmup_steps = total_optimization_steps*kwargs['warmup_proportion']
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_optimization_steps)
        # scheduler.verbose = True

    if kwargs["fp16"]:
        scaler = torch.cuda.amp.GradScaler()

    logger.info("******** Training ********")
    logger.info(f"    Num samples: {len(train_dataset)}")
    logger.info(f"    Num epochs: {kwargs['num_epochs']}")
    logger.info(f"    Batch size: {kwargs['effective_batch_size']}")
    logger.info(f"    Total optimization steps: {total_optimization_steps}")

    best_f1 = 0
    for epoch in range(kwargs['num_epochs']):
        logger.info(f"EPOCH: {epoch+1}")
        total_loss = 0
        optimizer.zero_grad()
        model.train()

        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for step, batch in pbar:
            batch = utils.batch_to_device(batch, kwargs['device'])
            input_ids, attention_mask, labels, samples = batch
            input_ids = input_ids.squeeze()
            attention_mask = attention_mask.squeeze()

            if kwargs['fp16']:
                with torch.cuda.amp.autocast():
                    per_sample_loss = model.calculate_loss(
                        input_ids, attention_mask, labels)
                    if kwargs['pos_sample_weight'] > 1:
                        sample_weight = labels*kwargs['pos_sample_weight']
                        sample_weight = torch.clamp(sample_weight, min=1.0)
                        per_sample_loss = per_sample_loss*sample_weight
                    loss = torch.sum(per_sample_loss)
                    loss = loss/gradient_accumulation_steps
                scaler.scale(loss).backward()
                total_loss += loss.item()

                if ((step + 1) % gradient_accumulation_steps) == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), kwargs["max_grad_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                    if kwargs['warmup_proportion'] > 0:
                        scheduler.step()
                    optimizer.zero_grad()
            else:
                per_sample_loss = model.calculate_loss(
                    input_ids, attention_mask, labels)
                if kwargs['pos_sample_weight'] > 1:
                    sample_weight = labels*kwargs['pos_sample_weight']
                    sample_weight = torch.clamp(sample_weight, min=1.0)
                    per_sample_loss = per_sample_loss*sample_weight
                loss = torch.sum(per_sample_loss)
                loss = loss/gradient_accumulation_steps
                loss.backward()
                total_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), kwargs["max_grad_norm"])
                if ((step + 1) % gradient_accumulation_steps) == 0:
                    optimizer.step()
                    if kwargs['warmup_proportion'] > 0:
                        scheduler.step()
                    optimizer.zero_grad()

            desc = f"TRAIN LOSS: {total_loss/(step+1):0.4f}"
            pbar.set_description(desc)

        tp, fp, fn, tn = 0, 0, 0, 0
        for batch in dev_dataloader:
            batch = utils.batch_to_device(batch, kwargs['device'])
            input_ids, attention_mask, labels, samples = batch
            input_ids = input_ids.squeeze()
            attention_mask = attention_mask.squeeze()
            with torch.no_grad():
                preds = model.predict(input_ids, attention_mask)
            for l, p in zip(labels.squeeze(), preds.squeeze()):
                if l == 1:
                    if p == 1:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if p == 1:
                        fp += 1
                    else:
                        tn += 1

        precision = tp/(tp+fp) if (tp+fp) > 0 else 0
        recall = tp/(tp+fn) if (tp+fn) > 0 else 1
        f1 = 2*precision*recall / \
            (precision+recall) if (precision+recall) > 0 else 0
        logger.info(f"**DEV**    TP: {tp} - FP: {fp} - FN: {fn} - TN: {tn}")
        logger.info(f"**DEV**    PR: {precision} - RE: {recall} - F1: {f1}")
        if f1 > best_f1:
            best_f1 = f1
            if kwargs['output_dir']:
                output_dir = os.path.join(
                    kwargs['output_dir'], f"F1-{best_f1:0.2f}")
                model.save_pretrained(output_dir)

    if kwargs['output_dir']:
        output_dir = os.path.join(
            kwargs['output_dir'], f"F1-{f1:0.2f}_final")
        model.save_pretrained(output_dir)


if __name__ == "__main__":
    args = utils.parse_args()
    main(**args)
