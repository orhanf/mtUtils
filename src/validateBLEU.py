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
        #self.device   = self.select_device()

    def select_device(self):
        '''
        this is a dummy way
        '''
        device='gpu0'
        try:
            os.environ["THEANO_FLAGS"] = "device=gpu0"
            import theano
        except:
            device='gpu2'
            try:
                os.environ["THEANO_FLAGS"] = "device=gpu0"
                import theano
            except:
                device='cpu'
        return device
    
    def call_bleu_script(self):
        '''
        open pipe and call the script
        '''
        try:
            p = subprocess.Popen([self.params['script'] + ' ' + 
                                  self.params['prefix'] + ' ' +
                                  self.params['base']   + ' ' +
                                  self.device + ' ' +
                                " | grep 'BLEU =\|CODEWORD ='"], stdout=subprocess.PIPE, shell=True)
            output = p.communicate()[0]
            tst_parse = re.search('(?<=Tst BLEU = )[.0-9]*', output)
            dev_parse = re.search('(?<=Tst BLEU = )[.0-9]*', output)
            cod_parse = re.search('(?<=CODEWORD = )[0-9]*', output)
            self.tstBLEU  = float(tst_parse.group())
            self.devBLEU  = float(dev_parse.group())
            self.codeword = int(cod_parse.group())
        except:
            print 'error in call_bleu_script()'
            
    def write_results(self):
        '''
        write results to output file, create if not exists
        '''
        if not os.path.isfile(params['outfile']+self.codeword):
            with open(params['outfile']+self.codeword, 'w+') as f: 
                f.write('CODEWORD\tTST_BLEU\tDEV_BLEU\n')
        with open(params['outfile']+self.codeword, 'a') as f: 
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

    params ={}
    params['script'] = sys.argv[1]
    params['prefix'] = sys.argv[2]
    params['base']   = sys.argv[3]
    params['outfile']= sys.argv[4]
    params['model']  = params['base'] + '_model.npz'
    
    signal.signal(signal.SIGINT, sigint_handler)
    
    controllerSleep = 60*30 # check every 30 minutes
    
    # this is for file access
    lock = threading.Lock()

    threads = []

    lastModified = os.path.getmtime(params['model'])
    threads = []
    while not shutdown:
        if os.path.getmtime(modelFilename) > lastModified:
            time.sleep(5) # wait for file transfers 
            print 'LAUNCHING THREAD {}'.format(len(threads))
            threads.append(CalculateBLEU(len(threads),lock, params))
            threads[-1].start()
        time.sleep(controllerSleep)
        
        
    print 'waiting for threads to join()...'
    [t.join() for t in threads]
    print 'done'
    