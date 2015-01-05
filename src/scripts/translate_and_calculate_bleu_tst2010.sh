#!/bin/bash

# change these
prefix=$1
trained_models_path=/data/lisatmp3/firatorh/nmt/tr-en_lm/trainedModels/union
ghog_nmt_path=~/git/GroundHog/experiments/nmt
source_file=/data/lisatmp3/firatorh/nmt/tr-en_lm/tst/IWSLT14.TED.tst2010.tr-en.tr.tok.seg
gold_file=/data/lisatmp3/firatorh/nmt/tr-en_lm/tst/IWSLT14.TED.tst2010.tr-en.en.tok
translation_file=./${1}.IWSLT14.TED.test2010.tr-en.TRANSLATION
scripts_path=/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts

# these are usually same
state_file=${trained_models_path}/${prefix}_state.pkl
model_file=${trained_models_path}/${prefix}_best_bleu_model.npz
sample_py=${ghog_nmt_path}/sample.py
eval_bleu_perl=${scripts_path}/multi-bleu.perl

# first get the translations
echo 'translating from english to turkish...'
THEANO_FLAGS='floatX=float32, device=gpu0' python $sample_py --beam-search --state $state_file $model_file --source $source_file --trans $translation_file --beam-size 20 

# calculate bleu score
echo 'calculating bleu score...'
perl $eval_bleu_perl  $gold_file < ${translation_file}

echo 'done!'



