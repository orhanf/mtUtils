#!usr/bin/python
'''
sample call:
    validateBLEU.py script.sh prefix basedir outfile
'''
import logging
import sys
import subprocess
import traceback

from Queue import Queue
from threading import Thread

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def get_params_cs_en_LV():
    '''
    parameters to change, filenames etc
    '''
    params = {}
    params['script'] = '/part/02/Tmp/firatorh/wmt15/trainedModels/evaluate_model.sh'
    #params['script'] = '~/evaluate_model.sh'
    params['modelIdx'] = [17]# range(1, 5)
    params['root_dir'] = '/part/02/Tmp/firatorh/wmt15/trainedModels'
    params['src_file'] = '/data/lisatmp3/jeasebas/nmt/data/wmt15/full/dev/tok/newstest2013.tok.cs'
    params['ref_file'] = '/data/lisatmp3/jeasebas/nmt/data/wmt15/full/dev/tok/newstest2013.tok.en'
    params['device'] = 'cpu'
    params['prefix'] = 'search_lv_cs_fi'
    params['beam_size'] = 12
    return params


class Worker(Thread):
    """Thread executing tasks from a given tasks queue"""
    def __init__(self, tasks):
        Thread.__init__(self)
        self.tasks = tasks
        self.daemon = True
        self.start()

    def run(self):
        while True:
            func, args, kargs = self.tasks.get()
            try:
                func(*args, **kargs)
            except Exception, e:
                print e
            finally:
                self.tasks.task_done()


class ThreadPool:
    """Pool of threads consuming tasks from a queue"""
    def __init__(self, num_threads):
        self.tasks = Queue(num_threads)
        for _ in range(num_threads):
            Worker(self.tasks)

    def add_task(self, func, *args, **kargs):
        """Add a task to the queue"""
        self.tasks.put((func, args, kargs))

    def wait_completion(self):
        """Wait for completion of all the tasks in the queue"""
        self.tasks.join()


def call_script(model_idx, params):

        try:
            logger.info('Thread for model{} is running'.format(model_idx))
            p = subprocess.Popen([params['script'] + ' ' +
                                  ' %d ' % model_idx +
                                  params['root_dir'] + ' ' +
                                  params['src_file'] + ' ' +
                                  params['ref_file'] + ' ' +
                                  params['device'] + ' ' +
                                  params['prefix'] + ' ' +
                                  ' %d ' % params['beam_size']],
                                 stdout=subprocess.PIPE, shell=True)
            output = p.communicate()[0]
            with open('model{}.out'.format(model_idx), 'w') as out_file:
                out_file.write(output)

        except:
            traceback.print_exc(file=sys.stdout)
            logger.info('error in call_bleu_script()')


if __name__ == "__main__":

    params = get_params_cs_en_LV()

    logger.info('script for bleu :{}'.format(params['script']))
    logger.info('model idx       :{}'.format(params['modelIdx']))
    logger.info('root directory  :{}'.format(params['root_dir']))
    logger.info('source file     :{}'.format(params['src_file']))
    logger.info('reference file  :{}'.format(params['ref_file']))
    logger.info('device          :{}'.format(params['device']))
    logger.info('prefix          :{}'.format(params['prefix']))
    logger.info('beam size       :{}'.format(params['beam_size']))

    pool = ThreadPool(2)

    for i, d in enumerate(params['modelIdx']):
        pool.add_task(call_script, d, params)

    logger.info('waiting for threads to join()...')
    pool.wait_completion()
    logger.info('done')
