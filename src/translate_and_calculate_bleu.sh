#/bin/bash

# arguments to script
PREFIX=$1
BASE_DIR=$2
DEVICE=$3

# unique id, time-stamp
CODEWORD=$(date +%s)

# script needs a directory structure 
# like ./trainedModels ./tst ./dev 
TRAINEDMODELS_PATH=${BASE_DIR}/trainedModels

# Groundhog path for sample.py
SAMPLE_PY=/u/firatorh/git/GroundHog/experiments/nmt/sample.py

# input and output files for test set
TST_SOURCE=${BASE_DIR}/tst/$4
TST_GOLD=${BASE_DIR}/tst/$5
TST_OUT=${BASE_DIR}/${PREFIX}_${CODEWORD}.IWSLT14.TED.tst2010.zh-en.TRANSLATION

# input and output files for development set
DEV_SOURCE=${BASE_DIR}/dev/$6
DEV_GOLD=${BASE_DIR}/dev/$7
DEV_OUT=${BASE_DIR}/${PREFIX}_${CODEWORD}.IWSLT14.TED.dev2010.zh-en.TRANSLATION

# joint input and output files
INP_FILE=${BASE_DIR}/${CODEWORD}_INPUT
OUT_FILE=${BASE_DIR}/${CODEWORD}_OUTPUT 
cat $TST_SOURCE $DEV_SOURCE > $INP_FILE

# get line numbers of test 
NUMLINES_TST=$(cat $TST_SOURCE | wc -l )

# these are usually same
REF_STATE=${TRAINEDMODELS_PATH}/${PREFIX}_state.pkl
REF_MODEL=${TRAINEDMODELS_PATH}/${PREFIX}_model.npz
STATE=${BASE_DIR}/${PREFIX}_${CODEWORD}_state.pkl
MODEL=${BASE_DIR}/${PREFIX}_${CODEWORD}_model.npz

# path to bleu score function
EVAL_BLEU=/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl

# copy original state and model file first
echo 'copying state and model files'
cp $REF_STATE $STATE
cp $REF_MODEL $MODEL

# next get the translations
echo 'translating from chinese to english...'
THEANO_FLAGS="floatX=float32, device=${DEVICE}" python $SAMPLE_PY --beam-search --state $STATE $MODEL --source $INP_FILE --trans $OUT_FILE --beam-size 20

# split output file back to test and dev files
split -l $NUMLINES_TST $OUT_FILE $OUT_FILE
mv ${OUT_FILE}aa $TST_OUT
mv ${OUT_FILE}ab $DEV_OUT
rm $OUT_FILE
rm $INP_FILE

# calculate bleu score
TST_BLEU=$(perl $EVAL_BLEU  $TST_GOLD < $TST_OUT | grep -oP '(?<=BLEU = )[.0-9]+')
echo 'Tst BLEU =' $TST_BLEU

# calculate bleu score
DEV_BLEU=$(perl $EVAL_BLEU  $DEV_GOLD < $DEV_OUT | grep -oP '(?<=BLEU = )[.0-9]+')
echo 'Dev BLEU =' $DEV_BLEU

# append scores to translation files
echo 'CODEWORD =' ${CODEWORD}  
mv ${TST_OUT} ${TST_OUT}-BLEU${TST_BLEU}
mv ${DEV_OUT} ${DEV_OUT}-BLEU${DEV_BLEU}



