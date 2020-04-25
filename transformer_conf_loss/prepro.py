# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer.

Preprocess the wiki-large-dataset.
'''

import os
import errno
import sentencepiece as spm
import re
from hparams import Hparams
import logging

logging.basicConfig(level=logging.INFO)

def prepro(hp):
    """Load raw data -> Preprocessing -> Segmenting with sentencepice
    hp: hyperparams. argparse.
    """
    logging.info("# Check if raw files exist")
    train1 = "/mnt/nfs/work1/cs696e/krajbhara/project/data-simplification/data-simplification/wikilarge/wiki.full.aner.ori.train.src"
    train2 = "/mnt/nfs/work1/cs696e/krajbhara/project/data-simplification/data-simplification/wikilarge/wiki.full.aner.ori.train.dst"
    eval1 = "/mnt/nfs/work1/cs696e/krajbhara/project/data-simplification/data-simplification/wikilarge/wiki.full.aner.ori.valid.src"
    eval2 = "/mnt/nfs/work1/cs696e/krajbhara/project/data-simplification/data-simplification/wikilarge/wiki.full.aner.ori.valid.dst"
    test1 = "/mnt/nfs/work1/cs696e/krajbhara/project/data-simplification/data-simplification/wikilarge/wiki.full.aner.ori.test.src"
    test2 = "/mnt/nfs/work1/cs696e/krajbhara/project/data-simplification/data-simplification/wikilarge/wiki.full.aner.ori.test.dst"
    for f in (train1, train2, eval1, eval2, test1, test2):
        if not os.path.isfile(f):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), f)

    logging.info("# Preprocessing")
    # train
    _prepro = lambda x:  [line.strip() for line in open(x, 'r',encoding="utf-8").read().split("\n")]
    prepro_train1, prepro_train2 = _prepro(train1), _prepro(train2)
    assert len(prepro_train1)==len(prepro_train2), "Check if train source and target files match."

    #eval
    prepro_eval1, prepro_eval2 = _prepro(eval1), _prepro(eval2)
    assert len(prepro_eval1) == len(prepro_eval2), "Check if eval source and target files match."

    # test
    prepro_test1, prepro_test2 = _prepro(test1), _prepro(test2)
    assert len(prepro_test1) == len(prepro_test2), "Check if test source and target files match."



    logging.info("# write preprocessed files to disk")
    os.makedirs("/mnt/nfs/work1/cs696e/krajbhara/project/data-simplification/data-simplification/wikilarge/prepro", exist_ok=True)
    def _write(sents, fname):
        with open(fname, 'w', encoding = "utf-8") as fout:
            fout.write("\n".join(sents))

    _write(prepro_train1, "/mnt/nfs/work1/cs696e/krajbhara/project/data-simplification/data-simplification/wikilarge/prepro/train.src")
    _write(prepro_train2, "/mnt/nfs/work1/cs696e/krajbhara/project/data-simplification/data-simplification/wikilarge/prepro/train.dst")
    _write(prepro_train1+prepro_train2, "/mnt/nfs/work1/cs696e/krajbhara/project/data-simplification/data-simplification/wikilarge/prepro/train")
    _write(prepro_eval1, "/mnt/nfs/work1/cs696e/krajbhara/project/data-simplification/data-simplification/wikilarge/prepro/eval.src")
    _write(prepro_eval2, "/mnt/nfs/work1/cs696e/krajbhara/project/data-simplification/data-simplification/wikilarge/prepro/eval.dst")
    _write(prepro_test1, "/mnt/nfs/work1/cs696e/krajbhara/project/data-simplification/data-simplification/wikilarge/prepro/test.src")
    _write(prepro_test2, "/mnt/nfs/work1/cs696e/krajbhara/project/data-simplification/data-simplification/wikilarge/prepro/test.dst")

    logging.info("# Train a joint BPE model with sentencepiece")
    os.makedirs("/mnt/nfs/work1/cs696e/krajbhara/project/data-simplification/data-simplification/wikilarge/segmented", exist_ok=True)
    train_path = "/mnt/nfs/work1/cs696e/krajbhara/project/data-simplification/data-simplification/wikilarge/prepro/train"
    model_path = '/mnt/nfs/work1/cs696e/krajbhara/project/data-simplification/data-simplification/wikilarge/segmented/bpe'
    train = '--input=' + train_path + ' --pad_id=0 --unk_id=1 \
             --bos_id=2 --eos_id=3\
             --model_prefix='+ model_path +' --vocab_size={} \
             --model_type=bpe'.format(hp.vocab_size)
    spm.SentencePieceTrainer.Train(train)

    logging.info("# Load trained bpe model")
    sp = spm.SentencePieceProcessor()
    sp.Load('/mnt/nfs/work1/cs696e/krajbhara/project/data-simplification/data-simplification/wikilarge/segmented/bpe.model')

    logging.info("# Segment")
    def _segment_and_write(sents, fname):
        with open(fname, "w", encoding="utf-8") as fout:
            for sent in sents:
                pieces = sp.EncodeAsPieces(sent)
                fout.write(" ".join(pieces) + "\n")

    _segment_and_write(prepro_train1, '/mnt/nfs/work1/cs696e/krajbhara/project/data-simplification/data-simplification/wikilarge/segmented/train.src.bpe')
    _segment_and_write(prepro_train2, '/mnt/nfs/work1/cs696e/krajbhara/project/data-simplification/data-simplification/wikilarge/segmented/train.dst.bpe')
    _segment_and_write(prepro_eval1, '/mnt/nfs/work1/cs696e/krajbhara/project/data-simplification/data-simplification/wikilarge/segmented/eval.src.bpe')
    _segment_and_write(prepro_eval2, '/mnt/nfs/work1/cs696e/krajbhara/project/data-simplification/data-simplification/wikilarge/segmented/eval.dst.bpe')
    _segment_and_write(prepro_test1, '/mnt/nfs/work1/cs696e/krajbhara/project/data-simplification/data-simplification/wikilarge/segmented/test.src.bpe')

    logging.info("Let's see how segmented data look like")

if __name__ == '__main__':
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()
    prepro(hp)
    logging.info("Done")