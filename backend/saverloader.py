import torch
import os
import hyperparams as hyp
import re
import ipdb 
st = ipdb.set_trace

def load_weights(model, optimizer):
    if hyp.total_init:
        print("TOTAL INIT")
        print(hyp.total_init)
        start_iter = load(hyp.total_init, model, optimizer)
        iter = start_iter
        if start_iter:
            print("loaded full model. resuming from iter %d" % start_iter)
        else:
            print("could not find a full model. starting from scratch")
    else:
        start_iter = 0
        inits = {"featnet": hyp.feat_init,
                 "viewnet": hyp.view_init,
                 "visnet": hyp.vis_init,
                 "flownet": hyp.flow_init,
                 "embnet2D": hyp.emb2D_init,
                 # "embnet3d": hyp.emb3D_init, # no params here really
                 'detnet': hyp.det_init,
                 'pixor': hyp.pixor_init,
                 'quantizer': hyp.quant_init,
                 "inpnet": hyp.inp_init,
                 "egonet": hyp.ego_init,
                 "occnet": hyp.occ_init,
                 "preoccnet": hyp.preocc_init,
                 "munitnet": hyp.munit_init,
                 "smoothnet": hyp.smoothnet_init
        }
        iter = 0
        for part, init in list(inits.items()):
            # st()
            if init:
                if part == 'smoothnet':
                    model_part = model.smoothnet
                elif part == 'munitnet':
                    model_part = model.munitnet
                elif part == 'featnet':
                    model_part = model.featnet
                elif part == 'viewnet':
                    model_part = model.viewnet
                elif part == 'occnet':
                    model_part = model.occnet
                elif part == 'quantizer':
                    model_part = model.quantizer
                elif part == 'pixor':
                    model_part = model.pixor                    
                elif part == 'detnet':
                    model_part = model.detnet
                elif part == 'preoccnet':
                    model_part = model.preoccnet
                elif part == 'embnet2D':
                    model_part = model.embnet2D
                elif part == 'flownet':
                    model_part = model.flownet
                else:
                    assert(False)

                iter = load_part(model_part, part, init)
                if iter:
                    print("loaded %s at iter %d" % (init, iter))
                else:
                    print("could not find a checkpoint for %s" % init)
    start_iter = iter
    # st()
    if hyp.reset_iter:
        start_iter = 0
    return start_iter


def save(model, checkpoint_dir, step, optimizer):
    digit_regex = re.compile('\d+')
    model_name = "model-%d.pth"%(step)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    path = os.path.join(checkpoint_dir, model_name)

    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, path)
    print("Saved a checkpoint: %s"%(path))
    
    # st()
    if hyp.delete_old_checkpoints:
        old_checkpoints = sorted([int(digit_regex.findall(old_model)[0]) for old_model in os.listdir(checkpoint_dir) if 'model' in old_model])
        if len(old_checkpoints) > hyp.delete_checkpoints_older_than:
            os.remove(os.path.join(checkpoint_dir, "model-%d.pth"%(old_checkpoints[0])))


def load(model_name, model, optimizer):
    print("reading full checkpoint...")
    checkpoint_dir = os.path.join("checkpoints/", model_name)
    step = 0
    if not os.path.exists(checkpoint_dir):
        print("...ain't no full checkpoint here!")
    else:
        ckpt_names = os.listdir(checkpoint_dir)
        steps = [int((i.split('-')[1]).split('.')[0]) for i in ckpt_names]
        if len(ckpt_names) > 0:
            step = max(steps)
            model_name = 'model-%d.pth'%(step)
            path = os.path.join(checkpoint_dir, model_name)
            print("...found checkpoint %s"%(path))

            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("...ain't no full checkpoint here!")
    return step


def load_part(model, part, init):
    print("reading %s checkpoint..." % part)
    init_dir = os.path.join("checkpoints", init)
    print(init_dir)
    step = 0
    if not os.path.exists(init_dir):
        print("...ain't no %s checkpoint here!"%(part))
    else:
        ckpt_names = os.listdir(init_dir)
        steps = [int((i.split('-')[1]).split('.')[0]) for i in ckpt_names]
        if len(ckpt_names) > 0:
            step = max(steps)
            model_name = 'model-%d.pth'%(step)
            path = os.path.join(init_dir, model_name)
            print("...found checkpoint %s"%(path),part)
            checkpoint = torch.load(path)
            model_state_dict = model.state_dict()
            # print(model_state_dict.keys())
            for load_para_name, para in checkpoint['model_state_dict'].items():
                model_para_name = load_para_name[len(part)+1:]
                # print(model_para_name, load_para_name)
                if hyp.no_bn:
                    if part+"."+model_para_name != load_para_name:
                        continue
                    else:
                        if model_para_name in model_state_dict.keys():
                            if part == "quantizer":
                                # st()
                                model_state_dict[model_para_name].copy_(para.data.reshape([hyp.object_quantize_dictsize*hyp.num_classes,-1]))
                            else:
                                try:
                                    model_state_dict[model_para_name].copy_(para.data)
                                except Exception as e:
                                    st()
                                    print("hello")
                else:
                    if part+"."+model_para_name != load_para_name:
                        continue
                    else:
                        model_state_dict[model_para_name].copy_(para.data)
                #model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            print("...ain't no %s checkpoint here!"%(part))
    # st()
    return step

