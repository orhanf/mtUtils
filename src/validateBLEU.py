#!usr/bin/python
'''
sample call:
    validateBLEU.py script.sh prefix basedir outfile
'''
import os
import sys
import numpy as np
import threading
import time
import shutil
import subprocess
import re
import signal
from time import sleep
import traceback

def get_params_zh_en():
    '''
    parameters to change, filenames etc
    '''
    params = {}
    params['script'] ='~/orhanf/mtUtils/src/translate_and_calculate_bleu.sh'
    params['prefix'] ='searchWithoutLM'
    params['base']   ='/data/lisatmp3/firatorh/nmt/zh-en_lm'
    params['outfile']='/data/lisatmp3/firatorh/nmt/zh-en_lm/searchWithoutLM_OUT'
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
    params['prefix'] ='searchWithoutLM'
    params['base']   ='/data/lisatmp3/firatorh/nmt/tr-en_lm'
    params['outfile']='/data/lisatmp3/firatorh/nmt/tr-en_lm/searchWithoutLM_OUT'
    params['tstSrc'] = 'IWSLT14.TED.tst2010.tr-en.tr.tok.seg'
    params['tstGld'] = 'IWSLT14.TED.tst2010.tr-en.en.tok'
    params['devSrc'] = 'IWSLT14.TED.dev2010.tr-en.tr.tok.seg'
    params['devGld'] = 'IWSLT14.TED.dev2010.tr-en.en.tok'
    params['model']  = params['base'] +'/trainedModels/'+\
                        params['prefix'] + '_model.npz'
    return params

class CalculateBLEU(threading.Thread):

    bestTstBLEU = 0.0
    bestDevBLEU = 0.0
    count = 0

    def __init__(self, tid, lock, params):
        """
        Constructor.
        """
        threading.Thread.__init__(self)
        self.tid      = tid
        self.lock     = lock
        self.params   = params
        self.tstBLEU  = 0.0
        self.devBLEU  = 0.0
        self.codeword = 'error'
        self.device   = self.select_device()

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
        return 'cpu'

    def call_bleu_script(self):
        '''
        open pipe and call the script
        '''
        try:
            print 'Thread-{} is using device={}'.format(self.tid,self.device)
            p = subprocess.Popen([self.params['script'] + ' ' +
                                  self.params['prefix'] + ' ' +
                                  self.params['base']   + ' ' +
                                  self.device + ' ' +
                                  self.params['tstSrc'] + ' ' +
                                  self.params['tstGld'] + ' ' +
                                  self.params['devSrc'] + ' ' +
                                  self.params['devGld'] + ' ' +
                                " | grep 'BLEU =\|CODEWORD ='"], stdout=subprocess.PIPE, shell=True)
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
        if not os.path.isfile(params['outfile']):
            with open(params['outfile'], 'w+') as f:
                f.write('CODEWORD TST_BLEU DEV_BLEU\n')
        with open(params['outfile'], 'a') as f:
            f.write('{}\t{}\t{}\n'.format(self.codeword,
                                          self.bestTstBLEU,
                                          self.bestDevBLEU))
    def run(self):
        """
        Thread run method. Calculate BLEU on Test and Dev sets.
        """
        self.call_bleu_script()

        self.lock.acquire()
        self.write_results()
        self.lock.release()

        print 'THREAD-{} done'.format(self.tid)

shutdown = False

def sigint_handler(_signo, _stack_frame):
    '''
    handles keyboard interrupt, ctrl+c
    '''
    print 'process interrupted'
    global shutdown
    shutdown = True

if __name__ == "__main__":

    params = get_params_zh_en()

    params['outfile'] += str(int(time.time()))

    signal.signal(signal.SIGINT, sigint_handler)

    controllerSleep = 60*30 # check every 30 minutes


    # this is for file access
    lock = threading.Lock()

    print 'script for bleu :{}'.format(params['script'])
    print 'model prefix    :{}'.format(params['prefix'])
    print 'base directory  :{}'.format(params['base'])
    print 'output filename :{}'.format(params['outfile'])
    print 'model filename  :{}'.format(params['model'])
    print 'tst source file :{}'.format(params['tstSrc'])
    print 'tst gold file   :{}'.format(params['tstGld'])
    print 'dev source file :{}'.format(params['devSrc'])
    print 'dev gold file   :{}'.format(params['devGld'])
    print 'controller sleep:{} minutes'.format(controllerSleep/60)

    lastModified = 0
    threads = []
    while not shutdown:
        currModified = os.path.getmtime(params['model'])
        if currModified > lastModified:
            time.sleep(5) # wait for file transfers
            print 'LAUNCHING THREAD {}'.format(len(threads))
            threads.append(CalculateBLEU(len(threads),lock, params))
            threads[-1].start()
            lastModified = currModified
        time.sleep(controllerSleep)

    print 'waiting for threads to join()...'
    [t.join() for t in threads]
    print 'done'

