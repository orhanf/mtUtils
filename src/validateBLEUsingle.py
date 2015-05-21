#!usr/bin/python
'''
sample call:
    validateBLEU.py script.sh prefix basedir outfile
'''
import os
import sys
import numpy as np
import time
import shutil
import subprocess
import re
import signal
import argparse
from time import sleep
import traceback

LIMIT_ITER=np.inf
SHUT_DOWN=False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plist", default="get_params_tr_en",
            help="Parameter list for bleu script")
    parser.add_argument("-i", "--nIter", type=int, default=LIMIT_ITER,
            help="Number of iterations")
    parser.add_argument("-d", "--device", type=str, default='gpu0',
            help="Device to use gpu0/gpu2/cpu")
    return parser.parse_args()

def get_params_zh_en():
    '''
    parameters to change, filenames etc
    '''
    params = {}
    params['script'] ='~/orhanf/mtUtils/src/translate_and_calculate_bleu.sh'
    params['prefix'] = 'searchWithLM0'
    params['base']   = '/data/lisatmp3/firatorh/nmt/zh-en_lm'
    params['outfile']= '/data/lisatmp3/firatorh/nmt/zh-en_lm/searchWithLM0_OUT'
    params['tstSrc'] = 'IWSLT14.TED.tst2010.zh-en.zh.xml.txt.trimmed'
    params['tstGld'] = 'IWSLT14.TED.tst2010.zh-en.en.tok'
    params['devSrc'] = 'IWSLT14.TED.dev2010.zh-en.zh.xml.txt.trimmed'
    params['devGld'] = 'IWSLT14.TED.dev2010.zh-en.en.tok'
    params['model']  = params['base'] +'/trainedModels/'+\
                        params['prefix'] + '_model.npz'
    return params

def get_params_tr_en():
    '''
    parameters to change, filenames etc
    '''
    params = {}
    params['script'] ='~/orhanf/mtUtils/src/translate_and_calculate_bleu.sh'
    params['prefix'] = 'searchWithLM0'
    params['base']   = '/data/lisatmp3/firatorh/nmt/tr-en_lm'
    params['outfile']='/data/lisatmp3/firatorh/nmt/tr-en_lm/searchWithLM0_OUT'
    params['tstSrc'] = 'IWSLT14.TED.tst2010.tr-en.tr.tok.seg'
    params['tstGld'] = 'IWSLT14.TED.tst2010.tr-en.en.tok'
    params['devSrc'] = 'IWSLT14.TED.dev2010.tr-en.tr.tok.seg'
    params['devGld'] = 'IWSLT14.TED.dev2010.tr-en.en.tok'
    params['model']  = params['base'] +'/trainedModels/'+\
                        params['prefix'] + '_model.npz'
    return params

def get_params_fi_en():
    '''
    parameters to change, filenames etc
    '''
    params = {}
    params['script'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/withoutLM/translate_and_calculate_bleu_TM_only.sh'
    params['prefix'] = 'refGHOG_adadelta_40k_reshuf'
    params['base']   = '/data/lisatmp3/firatorh/nmt/tr-en_lm'
    params['outfile']='/data/lisatmp3/firatorh/nmt/tr-en_lm/searchWithLM0_OUT'
    params['tstSrc'] = 'IWSLT14.TED.tst2010.tr-en.tr.tok.seg'
    params['tstGld'] = 'IWSLT14.TED.tst2010.tr-en.en.tok'
    params['devSrc'] = 'IWSLT14.TED.dev2010.tr-en.tr.tok.seg'
    params['devGld'] = 'IWSLT14.TED.dev2010.tr-en.en.tok'
    params['model']  = params['base'] +'/trainedModels/'+\
                        params['prefix'] + '_model.npz'
    return params

def get_params_tr_en_shallowLM():
    '''
    parameters to change, filenames etc
    '''
    params = {}
    params['script'] = '~/orhanf/mtUtils/src/translate_and_calculate_bleu_shallow.sh'
    params['prefix'] = 'searchWithoutLM_1417150916'
    params['base']   = '/data/lisatmp3/firatorh/nmt/tr-en_lm/outputs/searchWithoutLM'
    params['outfile']= '/data/lisatmp3/firatorh/nmt/shallow_fusion/tr-en_lm/searchWithShallowLM_OUT'
    params['tstSrc'] = 'IWSLT14.TED.tst2010.tr-en.tr.tok.seg'
    params['tstGld'] = 'IWSLT14.TED.tst2010.tr-en.en.tok'
    params['devSrc'] = 'IWSLT14.TED.dev2010.tr-en.tr.tok.seg'
    params['devGld'] = 'IWSLT14.TED.dev2010.tr-en.en.tok'
    params['model']  = params['base'] + '/' +\
                        params['prefix'] + '_model.npz'
    params['useAuxLM'] = True
    params['lmModel']  = '/data/lisatmp3/firatorh/nmt/en_lm/lm_model.npz'
    params['lmState']  = '/data/lisatmp3/firatorh/nmt/en_lm/lm_state.pkl'
    params['eta']      = 0.5
    return params

def get_params_zh_en_shallowLM():
    '''
    parameters to change, filenames etc
    '''
    params = {}
    params['script']='~/orhanf/mtUtils/src/translate_and_calculate_bleu_shallow.sh'
    params['prefix'] = 'searchWithshallowLM'
    params['base']   = '/data/lisatmp3/firatorh/nmt/zh-en_lm'
    params['outfile']= '/data/lisatmp3/firatorh/nmt/zh-en_lm/searchWithLM0_OUT'
    params['tstSrc'] = 'IWSLT14.TED.tst2010.zh-en.zh.xml.txt.trimmed'
    params['tstGld'] = 'IWSLT14.TED.tst2010.zh-en.en.tok'
    params['devSrc'] = 'IWSLT14.TED.dev2010.zh-en.zh.xml.txt.trimmed'
    params['devGld'] = 'IWSLT14.TED.dev2010.zh-en.en.tok'
    params['model']  = params['base'] +'/trainedModels/'+\
                        params['prefix'] + '_model.npz'
    params['useAuxLM'] = True
    params['lmModel']  ='/data/lisatmp3/firatorh/nmt/en_lm/lm_model.npz'
    params['lmState']  ='/data/lisatmp3/firatorh/nmt/en_lm/lm_state.pkl'
    params['eta']      = 0.5
    return params

class CalculateBLEU(object):

    bestTstBLEU = 0.0
    bestDevBLEU = 0.0
    count = 0
    bestBleuFilename = ''

    def __init__(self, tid, params):
        """
        Constructor.
        """
        self.tid      = tid
        self.params   = params
        self.tstBLEU  = 0.0
        self.devBLEU  = 0.0
        self.codeword = 'error'
        #self.device   = self.select_device()

    def select_device(self):
        '''
        this is a dummy way
        '''
        '''
        device='gpu0'
        try:
            os.environ["THEANO_FLAGS"] = "device=gpu0"
            import theano
        except:
            device='gpu2'
            try:
                os.environ["THEANO_FLAGS"] = "device=gpu2"
                import theano
            except:
                device='cpu'
        '''
        return 'gpu0'

    def call_bleu_script(self):
        '''
        open pipe and call the script
        '''
        try:
            print 'Job-{} is using device={}'.format(self.tid,params['device'])
            paramList = '{} {} {} {} {} {} {} {} '.format(self.params['script'],
                    self.params['script'],
                    self.params['base'],
                    self.params['device'],
                    self.params['tstSrc'],
                    self.params['tstGld'],
                    self.params['devSrc'],
                    self.params['devGld'])
            if 'useAuxLM' in self.params and self.params['useAuxLM']:
                paramList += ' {} {} {}'.format(self.params['lmModel'],
                        self.params['lmState'],self.params['eta'])

            p = subprocess.Popen([paramList +
                                " | grep 'BLEU =\|CODEWORD ='"],
                                stdout=subprocess.PIPE, shell=True)
            output = p.communicate()[0]
            tst_parse = re.search('(?<=Tst BLEU = )[.0-9]*', output)
            dev_parse = re.search('(?<=Dev BLEU = )[.0-9]*', output)
            cod_parse = re.search('(?<=CODEWORD = )[0-9]*', output)
            self.tstBLEU  = float(tst_parse.group())
            self.devBLEU  = float(dev_parse.group())
            self.codeword = int(cod_parse.group())
        except:
            traceback.print_exc(file=sys.stdout)
            print 'error in call_bleu_script()'

    def write_results(self):
        '''
        write results to output file, create if not exists
        '''
        with open(params['outfile'], 'a') as f:
            f.write('{}\t{}\t{}\n'.format(self.codeword,
                                          self.tstBLEU,
                                          self.devBLEU))
    def run(self):
        """
        Thread run method. Calculate BLEU On Test and Dev sets.
        """
        self.call_bleu_script()
        self.write_results()
        self.clean_files()
        print 'JOB-{} done'.format(self.tid)

    def clean_files(self):
        """
        Clean copied state and model files if necessary
        """
        stateThis = '{}/{}_{}_state.pkl'.format(self.params['base'], self.params['prefix'], self.codeword)
        modelThis = '{}/{}_{}_model.npz'.format(self.params['base'], self.params['prefix'], self.codeword)
        stateBest = '{}_state.pkl'.format(self.bestBleuFilename)
        modelBest = '{}_model.npz'.format(self.bestBleuFilename)

        if self.bestDevBLEU >= self.devBLEU:
            if os.path.exists(stateThis):
                os.remove(stateThis)
            if os.path.exists(modelThis):
                os.remove(modelThis)
        else:
            if os.path.exists(modelBest):
                os.remove(modelBest)
            if os.path.exists(stateBest):
                os.remove(stateBest)
        self.bestBleuFilename = '{}/{}_{}'.format(self.params['base'], self.params['prefix'], self.codeword)
        self.bestDevBLEU = self.devBLEU
        self.bestTstBLEU = self.tstBLEU

def sigint_handler(_signo, _stack_frame):
    '''
    handles keyboard interrupt, ctrl+c
    '''
    print 'process interrupted'
    global SHUT_DOWN
    SHUT_DOWN = True

if __name__ == "__main__":

    # get parameters dict
    args = parse_args()
    params = eval(args.plist)()

    # append unique time stamp to output file
    params['outfile'] += str(int(time.time()))
    params['device'] = args.device

    signal.signal(signal.SIGINT, sigint_handler)

    controllerSleep = 60*2 # check every 2 minutes

    print 'script for bleu :{}'.format(params['script'])
    print 'model prefix    :{}'.format(params['prefix'])
    print 'base directory  :{}'.format(params['base'])
    print 'output filename :{}'.format(params['outfile'])
    print 'model filename  :{}'.format(params['model'])
    print 'tst source file :{}'.format(params['tstSrc'])
    print 'tst gold file   :{}'.format(params['tstGld'])
    print 'dev source file :{}'.format(params['devSrc'])
    print 'dev gold file   :{}'.format(params['devGld'])
    if 'useAuxLM' in params and params['useAuxLM']:
        print 'lm model file   :{}'.format(params['lmModel'])
        print 'lm state file   :{}'.format(params['lmState'])
        print 'balancing eta   :{}'.format(params['eta'])
    print 'controller sleep:{} minutes'.format(controllerSleep/60)

    # initialize output file and write the header
    if not os.path.isfile(params['outfile']):
        with open(params['outfile'], 'w+') as f:
            f.write('CODEWORD TST_BLEU DEV_BLEU\n')

    lastModified = 0
    counter = 0
    while not SHUT_DOWN and counter < args.nIter:
        currModified = os.path.getmtime(params['model'])
        if currModified > lastModified:
            time.sleep(5) # wait for file transfers
            print 'LAUNCHING JOB {}'.format(counter)
            CalculateBLEU(counter, params).run()
            lastModified = currModified
            counter += 1
        time.sleep(controllerSleep)

    print 'done'

