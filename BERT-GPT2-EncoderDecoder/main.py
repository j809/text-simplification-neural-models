import argparse
import json
import os
import random
import sys

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AdamW, AutoModel, PreTrainedEncoderDecoder, get_linear_schedule_with_warmup

from src.data import WikiDataset
from src.models import get_model
from src.train import BERT_GPT2_Trainer
from src.utils import FitResult, get_max_len
import math
from torch.utils.tensorboard import SummaryWriter
from src.models import BERT_GPT2_EncoderDecoder


def run_experiment(run_name, out_dir='./results', data_dir_prefix, model_path=None,
                   model_name='bert-base-uncased', model_type='encode-decode', decoder_model_name='gpt2', seed=None,
                   drive=False, do_test=True,
                   # Training params
                   bs_train=32, bs_test=None, batches=100, epochs=100,
                   early_stopping=3, checkpoints=None, lr=0.0005, reg=1e-3, max_len=0,
                   # Model params
                   beta1=0.9, beta2=0.999, epsilon=1e-6, weight_decay=0.0,
                   **kw):
    if not seed:
        seed = random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    if not bs_test:
        bs_test = max([bs_train // 4, 1])
    cfg = locals()

    tf = torchvision.transforms.ToTensor()

    with open(data_dir_prefix + '_test_source_file') as test_lines_file:
        test_lines = test_lines_file.readlines()
    with open(data_dir_prefix + '_train_source_file') as train_lines_file:
        train_lines = train_lines_file.readlines()
    with open(data_dir_prefix + '_val_source_file') as val_lines_file:
        val_lines = val_lines_file.readlines()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    size_test = batches * bs_test if batches > 0 else -1
    size_train = batches * bs_train if batches > 0 else -1

    if max_len == 0:
        #  max_len = get_max_len(test_lines[:size_test] + train_lines[:size_train] + val_lines[:size_test] + hard_test_lines[:size_test],
        max_len = get_max_len(test_lines[:size_test] + train_lines[:size_train] + val_lines[:size_test],
                              '\t', tokenizer)
        print(f'Longest Sequence is: {max_len} token_ids')
        max_len = math.pow(2, math.ceil(math.log(max_len, 2)))
    max_len = int(max_len)

    print(f'Setting max_len to: {max_len}')

    ds_test = WikiDataset(test_lines, tokenizer, max_len=max_len)
    ds_val = WikiDataset(val_lines, tokenizer, max_len=max_len)
    ds_train = WikiDataset(train_lines, tokenizer, max_len=max_len)

    if batches > 0:
        ds_test = Subset(ds_test, range(batches * bs_test))
        ds_val = Subset(ds_val, range(batches * bs_test))
        ds_train = Subset(ds_train, range(batches * bs_train))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(tokenizer=tokenizer, model=model_type, model_name=model_name,
                      model_name_decoder=decoder_model_name, model_path=model_path)

    dl_train = torch.utils.data.DataLoader(ds_train, bs_train, shuffle=False)
    dl_val = torch.utils.data.DataLoader(ds_val, bs_test, shuffle=False)
    dl_test = torch.utils.data.DataLoader(ds_test, bs_test, shuffle=False)
    
    optimizer = AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), eps=epsilon, weight_decay=weight_decay)
    num_steps = batches if batches > 0 else len(dl_train) // bs_train
    num_steps = epochs * num_steps
    print(f"Number of training steps: {num_steps}")
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=round(0.1 * num_steps), num_training_steps=num_steps)
    writer = None
    if checkpoints is None:
        writer = SummaryWriter()

    trainerClass = BERT_GPT2_Trainer

    trainer = trainerClass(model, tokenizer, None, optimizer, scheduler, max_len, device)
    fit_res = trainer.fit(dl_train, dl_val, num_epochs=epochs, early_stopping=early_stopping, checkpoints=checkpoints, writer=writer)
    save_experiment(model, run_name, out_dir, cfg, fit_res)

def gen_model():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    decoder_tokenizer = AutoTokenizer.from_pretrained('gpt2')

    model = BERT_GPT2_EncoderDecoder('bert-base-uncased', 'gpt2', [tokenizer, decoder_tokenizer])
    model = torch.load('checkpoint')
    model.eval()
    model.to('cuda')

    with open('wiki/wiki_pair_test_source_file') as test_lines_file:
        test_lines = test_lines_file.readlines()
    
    for l in test_lines:
        splits = l.split('\t')

        encode_dict = tokenizer.encode_plus('[CLS] ' + splits[0], max_length=64, pad_to_max_length=True)
        indexed_tokens = encode_dict['input_ids']
        encoder_attention_mask = encode_dict['attention_mask']
        encoder_tokens_tensor = torch.tensor([indexed_tokens])
        encoder_tokens_tensor = encoder_tokens_tensor.to('cuda')

        decoder_input = ""

        decoder_dict = tokenizer.encode_plus(decoder_input, max_length=64, pad_to_max_length=True)
        tokenized_decoder_input = decoder_dict['input_ids']
        decoder_attention_mask = decoder_dict['attention_mask']

        decoder_tokens_tensor = torch.tensor([tokenized_decoder_input])
        decoder_tokens_tensor = decoder_tokens_tensor.to('cuda')

        predicted_ids = model.decoder.generate(input_ids=encoder_tokens_tensor,
                    max_length=64,
                    do_sample=True,
                    temperature=0.7,
                    repetition_penalty=1.2)
        predicted_text = tokenizer.decode(predicted_ids[0].tolist(), clean_up_tokenization_spaces=True)
            
        print(predicted_text)


def save_experiment(model, run_name, out_dir, config, fit_res):
    output = dict(
        config=config,
        results=fit_res._asdict()
    )
    output_filename = f'{os.path.join(out_dir, run_name)}.json'
    os.makedirs(out_dir, exist_ok=True)
    with open(output_filename, 'w') as f:
        json.dump(output, f, indent=2)

    torch.save(model, 'checkpoint')
    print(f'*** Output file {output_filename} written')


def load_experiment(filename):
    with open(filename, 'r') as f:
        output = json.load(f)

    config = output['config']
    fit_res = FitResult(**output['results'])

    return config, fit_res


def parse_cli():
    p = argparse.ArgumentParser(description='Experiments')
    sp = p.add_subparsers(help='Sub-commands')

    # Experiment config
    sp_exp = sp.add_parser('run-exp', help='Run experiment with a single '
                                           'configuration')
    sp_exp.set_defaults(subcmd_fn=run_experiment)
    sp_exp.add_argument('--run-name', '-n', type=str,
                        help='Name of run and output file', required=True)
    sp_exp.add_argument('--out-dir', '-o', type=str, help='Output folder',
                        default='./results', required=False)
    sp_exp.add_argument('--seed', '-s', type=int, help='Random seed',
                        default=None, required=False)
    sp_exp.add_argument('--drive', '-d', type=bool, help='Pass "True" if you are running this on Google Colab',
                        default=False, required=False)
    sp_exp.add_argument('--do-test', '-t', type=bool, help='Pass "True" if you want to run a test on test set',
                        default=True, required=False)                    

    # # Training
    sp_exp.add_argument('--bs-train', type=int, help='Train batch size',
                        default=128, metavar='BATCH_SIZE')
    sp_exp.add_argument('--bs-test', type=int, help='Test batch size',
                        metavar='BATCH_SIZE')
    sp_exp.add_argument('--batches', type=int,
                        help='Number of batches per epoch', default=100)
    sp_exp.add_argument('--epochs', type=int,
                        help='Maximal number of epochs', default=100)
    sp_exp.add_argument('--early-stopping', type=int,
                        help='Stop after this many epochs without '
                             'improvement', default=3)
    sp_exp.add_argument('--checkpoints', type=str,
                        help='Save model checkpoints to this file when test '
                             'accuracy improves', default=None)
    sp_exp.add_argument('--lr', type=float,
                        help='Learning rate', default=0.001)
    sp_exp.add_argument('--reg', type=float,
                        help='L2 regularization', default=1e-3)
    sp_exp.add_argument('--data-dir-prefix', type=str,
                        help='Prefix of the path to data', default='./data/snli_1.0/cl_snli')
    sp_exp.add_argument('--max-len', type=int,
                        help='Length of longest sequence (or bigger), 0 if you don\'t know', default=0)

    # # Model
    sp_exp.add_argument('--model-path', type=str,
                        help='Path to fined-tuned model', default=None)
    sp_exp.add_argument('--model-name', type=str,
                        help='Name of the huggingface model', default='bert-base-uncased')
    sp_exp.add_argument('--model-type', type=str,
                        help='Type of the model (encode-decode or hybrid)', default='encode-decode')
    sp_exp.add_argument('--decoder-model-name', type=str,
                        help='Only if model type is hybrid', default='gpt2')
    sp_exp.add_argument('--beta1', '-b1', type=float,
                        default=0.9)
    sp_exp.add_argument('--beta2', '-b2', type=float,
                        default=0.999)
    sp_exp.add_argument('--epsilon', '-eps', type=float,
                        default=1e-6)
    sp_exp.add_argument('--weight-decay', '-wd', type=float,
                        default=0.0)
    # sp_exp.add_argument('--hidden-dims', '-H', type=int, nargs='+',
    #                     help='Output size of hidden linear layers',
    #                     metavar='H', required=True)
    # sp_exp.add_argument('--ycn', action='store_true', default=False,
    #                     help='Whether to use your custom network')

    # # Training
    sp_test.add_argument('--bs-test', type=int, help='Test batch size',
                         metavar='BATCH_SIZE')
    sp_test.add_argument('--batches', type=int,
                         help='Number of batches per epoch, pass "0" if you want the full database', default=100)
    sp_test.add_argument('--data-dir-prefix', type=str,
                         help='Prefix of the path to data', default='./data/snli_1.0/cl_snli')
    sp_test.add_argument('--max-len', type=int,
                         help='Length of longest sequence (or bigger), 0 if you don\'t know', default=0)

    # # Model
    sp_test.add_argument('--model-name', type=str,
                         help='Name of the huggingface model', default='bert-base-uncased')
    sp_test.add_argument('--model-path', type=str,
                         help='Path to fined-tuned model', default=None)
    sp_test.add_argument('--model-type', type=str,
                         help='Type of the model (encode-decode or hybrid)', default='encode-decode')
    sp_test.add_argument('--checkpoints', type=str,
                         help='Checkpoint to torch model', default=None)
    sp_test.add_argument('--decoder-model-name', type=str,
                         help='Only if model type is hybrid', default='gpt2')

    # # Generate using the model
    sp_gen = sp.add_parser('gen', help='Gen using the model')
    sp_gen.set_defaults(subcmd_fn=gen_model)

    parsed = p.parse_args()

    if 'subcmd_fn' not in parsed:
        p.print_help()
        sys.exit()
    return parsed


if __name__ == '__main__':
    parsed_args = parse_cli()
    subcmd_fn = parsed_args.subcmd_fn
    del parsed_args.subcmd_fn
    print(f'*** Starting {subcmd_fn.__name__} with config:\n{parsed_args}')
    subcmd_fn(**vars(parsed_args))
