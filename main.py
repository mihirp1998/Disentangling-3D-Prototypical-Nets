import click #argparse is behaving weirdly
import os
import cProfile
import logging
import ipdb 
import torch
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
st = ipdb.set_trace
logger = logging.Logger('catch_all')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["LC_ALL"]= 'C.UTF-8'

@click.command()
@click.argument("mode", required=True)
@click.option("--exp_name","--en", default="trainer_basic", help="execute expriment name defined in config")
@click.option("--run_name","--rn", default="1", help="run name")


def main(mode, exp_name, run_name):
    if mode:
        if "cs" == mode:
            mode = "CLEVR_STA"
        elif "nel" == mode:
            mode = "NEL_STA"
        elif "style" == mode:
            mode = "STYLE_STA"
    
    if run_name == "1":
        run_name = exp_name

    os.environ["MODE"] = mode
    os.environ["exp_name"] = exp_name
    os.environ["run_name"] = run_name
    import hyperparams as hyp
    from model_clevr_sta import CLEVR_STA

    checkpoint_dir_ = os.path.join("checkpoints", hyp.name)


    if hyp.do_style_sta:
        log_dir_ = os.path.join("logs_style_sta", hyp.name)    
    elif hyp.do_clevr_sta:
        log_dir_ = os.path.join("logs_clevr_sta", hyp.name)    
    elif hyp.do_nel_sta:
        log_dir_ = os.path.join("logs_nel_sta", hyp.name)
    elif hyp.do_carla_sta:
        log_dir_ = os.path.join("logs_carla_sta", hyp.name)
    elif hyp.do_carla_flo:
        log_dir_ = os.path.join("logs_carla_flo", hyp.name)
    elif hyp.do_carla_obj:
        log_dir_ = os.path.join("logs_carla_obj", hyp.name)
    else:
        assert(False) # what mode is this?

    if not os.path.exists(checkpoint_dir_):
        os.makedirs(checkpoint_dir_)
    if not os.path.exists(log_dir_):
        os.makedirs(log_dir_)
    # st()
    try:
        if hyp.do_style_sta:
            model = STYLE_STA(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()

        elif hyp.do_clevr_sta:
            model = CLEVR_STA(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_nel_sta:
            model = NEL_STA(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_sta:
            model = CARLA_STA(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_flo:
            model = CARLA_FLO(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_obj:
            model = CARLA_OBJ(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        else:
            assert(False) # what mode is this?

    except (Exception, KeyboardInterrupt) as ex:
        logger.error(ex, exc_info=True)
        st()
        log_cleanup(log_dir_)

def log_cleanup(log_dir_):
    log_dirs = []
    for set_name in hyp.set_names:
        log_dirs.append(log_dir_ + '/' + set_name)

    for log_dir in log_dirs:
        for r, d, f in os.walk(log_dir):
            for file_dir in f:
                file_dir = os.path.join(log_dir, file_dir)
                file_size = os.stat(file_dir).st_size
                if file_size == 0:
                    os.remove(file_dir)

if __name__ == '__main__':
    main()
# cProfile.run('main()')