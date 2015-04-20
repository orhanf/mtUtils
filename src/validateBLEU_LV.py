#!usr/bin/python
import argparse
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
    params['modelIdx'] = range(1, 5)
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


parser = argparse.ArgumentParser()
parser.add_argument("--pool-size", type=int, default=2)
parser.add_argument("--proto", default="get_params_cs_en_LV",
        help="Parameter list for bleu script")
args = parser.parse_args()


if __name__ == "__main__":

    params = eval(args.proto)()

    for key, value in params.iteritems():
        logger.info('{:15} :{}'.format(key, value))

    pool = ThreadPool(args.pool_size)

    for i, d in enumerate(params['modelIdx']):
        pool.add_task(call_script, d, params)

    logger.info('waiting for threads to join()...')
    pool.wait_completion()
    logger.info('done')
