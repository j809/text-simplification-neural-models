#! /bin/sh
#
# run.sh
# Copyright (C) 2020 jay <jpanchal@umass.edu>
#
# Distributed under terms of the MIT license.
#

python main.py run-exp \
--run-name wiki --data-dir-prefix './wiki/wiki_pair' \
--bs-train 16 \
--batches 100 \
--bs-test 4 \
--model-name='bert-base-uncased' \
--lr 2e-5 \
--early-stopping 100 \
