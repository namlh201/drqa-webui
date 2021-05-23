#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# This code borrowed from https://github.com/facebookresearch/DrQA.git

import torch
import os

from .drqa_transformers_service import DrQATransformersService
from .pyserini_transformers_service import PyseriniTransformersService

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

drqa_data_directory = '../data'

config = {
    'reader-model': 'distilbert-base-cased-distilled-squad',
    'use-fast-tokenizer': True,
    'retriever-model': os.path.join(drqa_data_directory, 'wikipedia_using/en', 'docs-tfidf-ngram=1-hash=16777216-tokenizer=spacy.npz'),
    'doc-db': os.path.join(drqa_data_directory, 'wikipedia_using/en', 'docs.db'),
    'index-path': os.path.join(drqa_data_directory, 'index', 'lucene-index.enwiki-20180701-paragraphs'),
    'group-length': 500,
    'batch-size': 32,
    'num-workers': 2,
    'cuda': False,
    'gpu': 0
}

cuda = torch.cuda.is_available() and not config.get('no-cuda', False)
if cuda:
    torch.cuda.set_device(config.get('gpu', 0))
    logger.info('CUDA enabled (GPU %d)' % config.get('gpu', 0))
else:
    logger.info('Running on CPU only.')

logger.info('Initializing pipeline...')

class Services:
    def __init__(self, retriever='tfidf'):
        if retriever == 'tfidf':
            self.DrQA = DrQATransformersService(config)
        elif retriever == 'serini-bm25':
            self.DrQA = PyseriniTransformersService(config)


    def process(self, question, top_n=1, n_docs=5):
        return self.DrQA.process(question, top_n, n_docs)