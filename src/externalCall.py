#!usr/bin/python
import os
import sys
import numpy as np
import time
import subprocess
import signal
import argparse
import traceback

LIMIT_ITER = np.inf
SHUT_DOWN = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--params", default="get_params_fi_en_tmOnly_1",
                        help="Parameter list for bleu script")
    parser.add_argument("-i", "--nIter", type=int, default=LIMIT_ITER,
                        help="Number of iterations")
    parser.add_argument("-d", "--device", type=str, default='gpu0',
                        help="Device to use gpu0/gpu2/cpu")
    return parser.parse_args()


def get_params_fi_en_tmOnly_all():
    '''
    parameters to change, filenames etc
    '''
    params = {}
    params['script'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/translations/call_calculateBLEU_all.sh'
    params['prefix'] = 'refGHOG_adadelta_40k_reshuf'
    params['state'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/withoutLM/refGHOG_adadelta_40k_reshuf_state.pkl'
    params['model'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/withoutLM/refGHOG_adadelta_40k_reshuf_best_bleu_model.npz'
    params['outdir'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/translations/tmOnly/' + params['prefix']
    params['cmd_to_call'] = "{script} {device} {prefix} {state} {model} {outdir}".format(
        script=params['script'], device='{device}', prefix=params['prefix'],
        state=params['state'], model=params['model'], outdir=params['outdir'])
    return params

def get_params_fi_en_deep_all_fused_GHOG_adadelta_40k_cont():
    '''
    parameters to change, filenames etc
    '''
    params = {}
    params['script'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/translations/call_calculateBLEU_all.sh'
    params['prefix'] = 'fused_GHOG_adadelta_40k_cont'
    params['state'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/fused_GHOG_adadelta_40k_cont_state.pkl'
    params['model'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/fused_GHOG_adadelta_40k_cont_best_bleu_model.npz'
    params['outdir'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/translations/deep/' + params['prefix']
    params['cmd_to_call'] = "{script} {device} {prefix} {state} {model} {outdir}".format(
        script=params['script'], device='{device}', prefix=params['prefix'],
        state=params['state'], model=params['model'], outdir=params['outdir'])
    return params


def get_params_fi_en_deep_all_fused_GHOG_adadelta_40k():
    '''
    parameters to change, filenames etc
    '''
    params = {}
    params['script'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/translations/call_calculateBLEU_all.sh'
    params['prefix'] = 'fused_GHOG_adadelta_40k'
    params['state'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/fused_GHOG_adadelta_40k_state.pkl'
    params['model'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/fused_GHOG_adadelta_40k_best_bleu_model.npz'
    params['outdir'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/translations/deep/' + params['prefix']
    params['cmd_to_call'] = "{script} {device} {prefix} {state} {model} {outdir}".format(
        script=params['script'], device='{device}', prefix=params['prefix'],
        state=params['state'], model=params['model'], outdir=params['outdir'])
    return params


def get_params_fi_en_deep_all_fused_GHOG_adadelta_40k_reshuf0():
    '''
    parameters to change, filenames etc
    '''
    params = {}
    params['script'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/translations/call_calculateBLEU_all.sh'
    params['prefix'] = 'fused_GHOG_adadelta_40k_reshuf0'
    params['state'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/fused_GHOG_adadelta_40k_reshuf0_state.pkl'
    params['model'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/fused_GHOG_adadelta_40k_reshuf0_best_bleu_model.npz'
    params['outdir'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/translations/deep/' + params['prefix']
    params['cmd_to_call'] = "{script} {device} {prefix} {state} {model} {outdir}".format(
        script=params['script'], device='{device}', prefix=params['prefix'],
        state=params['state'], model=params['model'], outdir=params['outdir'])
    return params


def get_params_fi_en_deep_all_fused_GHOG_adadelta_40k_reshuf0_tmCont():
    '''
    parameters to change, filenames etc
    '''
    params = {}
    params['script'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/translations/call_calculateBLEU_all.sh'
    params['prefix'] = 'fused_GHOG_adadelta_40k_reshuf0_tmCont'
    params['state'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/fused_GHOG_adadelta_40k_reshuf0_tmCont_state.pkl'
    params['model'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/fused_GHOG_adadelta_40k_reshuf0_tmCont_best_bleu_model.npz'
    params['outdir'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/translations/deep/' + params['prefix']
    params['cmd_to_call'] = "{script} {device} {prefix} {state} {model} {outdir}".format(
        script=params['script'], device='{device}', prefix=params['prefix'],
        state=params['state'], model=params['model'], outdir=params['outdir'])
    return params


def get_params_fi_en_deep_all_fused_GHOG_adadelta_40k_reshuf1():
    '''
    parameters to change, filenames etc
    '''
    params = {}
    params['script'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/translations/call_calculateBLEU_all.sh'
    params['prefix'] = 'fused_GHOG_adadelta_40k_reshuf1'
    params['state'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/fused_GHOG_adadelta_40k_reshuf1_state.pkl'
    params['model'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/fused_GHOG_adadelta_40k_reshuf1_best_bleu_model.npz'
    params['outdir'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/translations/deep/' + params['prefix']
    params['cmd_to_call'] = "{script} {device} {prefix} {state} {model} {outdir}".format(
        script=params['script'], device='{device}', prefix=params['prefix'],
        state=params['state'], model=params['model'], outdir=params['outdir'])
    return params


def get_params_fi_en_deep_all_fused_GHOG_adadelta_40k_vecLMcont_reshuf1_noNoise():
    '''
    parameters to change, filenames etc
    '''
    params = {}
    params['script'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/translations/call_calculateBLEU_all.sh'
    params['prefix'] = 'fused_GHOG_adadelta_40k_vecLMcont_reshuf1_noNoise'
    params['state'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/fused_GHOG_adadelta_40k_vecLMcont_reshuf1_noNoise_state.pkl'
    params['model'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/fused_GHOG_adadelta_40k_vecLMcont_reshuf1_noNoise_model.npz'
    params['outdir'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/translations/deep/' + params['prefix']
    params['cmd_to_call'] = "{script} {device} {prefix} {state} {model} {outdir}".format(
        script=params['script'], device='{device}', prefix=params['prefix'],
        state=params['state'], model=params['model'], outdir=params['outdir'])
    return params


def get_params_fi_en_tmOnly_1():
    '''
    parameters to change, filenames etc
    '''
    params = {}
    params['script'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/translations/call_calculateBLEU_1.sh'
    params['prefix'] = 'refGHOG_adadelta_40k'
    params['state'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/withoutLM/refGHOG_adadelta_40k_state.pkl'
    params['model'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/withoutLM/refGHOG_adadelta_40k_best_bleu_model.npz'
    params['outdir'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/translations/tmOnly'
    params['cmd_to_call'] = "{script} {device} {prefix} {state} {model} {outdir}".format(
        script=params['script'], device='{device}', prefix=params['prefix'],
        state=params['state'], model=params['model'], outdir=params['outdir'])
    return params

def get_params_fi_en_tmOnly_2():
    '''
    parameters to change, filenames etc
    '''
    params = {}
    params['script'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/translations/call_calculateBLEU_2.sh'
    params['prefix'] = 'refGHOG_adadelta_40k'
    params['state'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/withoutLM/refGHOG_adadelta_40k_state.pkl'
    params['model'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/withoutLM/refGHOG_adadelta_40k_best_bleu_model.npz'
    params['outdir'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/translations/tmOnly'
    params['cmd_to_call'] = "{script} {device} {prefix} {state} {model} {outdir}".format(
        script=params['script'], device='{device}', prefix=params['prefix'],
        state=params['state'], model=params['model'], outdir=params['outdir'])
    return params

def get_params_fi_en_vectorLM_1():
    '''
    parameters to change, filenames etc
    '''
    params = {}
    params['script'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/translations/call_calculateBLEU_1.sh'
    params['prefix'] = 'fused_GHOG_adadelta_40k_vecLMcont_reshuf1_noNoise'
    params['state'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/fused_GHOG_adadelta_40k_vecLMcont_reshuf1_noNoise_state.pkl'
    params['model'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/deepFusion/fused_GHOG_adadelta_40k_vecLMcont_reshuf1_noNoise_model.npz'
    params['outdir'] = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/translations/deep'
    params['cmd_to_call'] = "{script} {device} {prefix} {state} {model} {outdir}".format(
        script=params['script'], device='{device}', prefix=params['prefix'],
        state=params['state'], model=params['model'], outdir=params['outdir'])
    return params

def call_script(tid, params):
    '''
    open pipe and call the script
    '''
    try:
        print 'Job-{} is using device={}'.format(tid, params['device'])
        cmd = params['cmd_to_call'].format(device=params['device'])
        subprocess.call(cmd, shell=True)
    except:
        traceback.print_exc(file=sys.stdout)
        print 'error in call_bleu_script()'


def sigint_handler(_signo, _stack_frame):
    '''
    handles keyboard interrupt, ctrl+c
    '''
    print 'process interrupted'
    global SHUT_DOWN
    SHUT_DOWN = True


if __name__ == "__main__":

    args = parse_args()
    params = eval(args.params)()
    params['device'] = args.device

    signal.signal(signal.SIGINT, sigint_handler)

    controllerSleep = 60 # check every 1 minutes

    for key, val in params.iteritems():
        print '{:15} :{}'.format(key, val)
    print '{:15} :{}'.format('nIter',args.nIter)
    print 'controller sleep:{} minutes'.format(controllerSleep/60)

    lastModified = 0
    counter = 0
    while not SHUT_DOWN and counter < args.nIter:
        if counter:
            time.sleep(controllerSleep)
        currModified = os.path.getmtime(params['model'])
        if currModified > lastModified:
            time.sleep(5) # wait for file transfers
            print 'LAUNCHING JOB {}'.format(counter)
            call_script(counter, params)
            lastModified = currModified
            counter += 1
    print 'done'

