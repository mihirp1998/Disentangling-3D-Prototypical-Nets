from exp_base import *
import os
import pretrained_nets_carla as pret
import ipdb
st = ipdb.set_trace
# THIS FILE IS FOR STORING STANDARD EXPERIMENTS/BASELINES FOR CARLA_STA MODE
############## choose an experiment ##############

# current = 'builder'
# current = 'trainer_basic'
current = '{}'.format(os.environ["exp_name"])


mod = '"{}"'.format(os.environ["run_name"]) # debug






exps['trainer_rgb_no_bn_munit_simple_cross_0.1_dsn_tmp'] = [
    'clevr_sta', #mode
    'clevr_shapes_vqa_highres_singleobj_test', # dataset
    '200k_iters',
    'lr4',
    'B2',
    'train_feat',
    'train_occ',
    'train_view',
    'pretrained_feat',
    'pretrained_view',
    'pretrained_occ',    
    'pretrained_munit',    
    'do_style_content',
    'reset_iter',
    'eval_boxes',    
    'train_munit',
    'smoothness_with_noloss',
    'no_bn',
    'fastest_logging',
    'munit_loss_term_wt_0.1',
    'basic_view_loss',
    # 'style_view_loss',
    'cycle_style_view_loss',
    'simple_adaingen',
]




exps['trainer_rgb_no_bn_munit_simple_cross_fewshot_test'] = [
    'clevr_sta', #mode
    'clevr_shapes_vqa_highres_multiobj_test', # dataset
    '200k_iters',
    'lr4',
    'B2',
    'train_feat',
    'pretrained_feat',
    'pretrained_munit',    
    'do_style_content',
    'reset_iter',
    'eval_boxes',    
    'train_munit',
    'run_few_shot_on_munit',
    'do_munit_fewshot',
    # 'smoothness_with_noloss',
    # 'avg_3d',
    'avg_3d',
    'no_bn',
    'fast_logging',
    'frozen_feat',
    'munit_loss_term_wt_0.1',
    'basic_view_loss',
    'break_constraint',
    # 'style_view_loss',
    # 'cycle_style_view_loss',
    'simple_adaingen',
]







exps['trainer_rgb_no_bn_munit_simple_cross_0.1_dsn'] = [
    'clevr_sta', #mode
    'clevr_shapes_vqa_highres_singleobj', # dataset
    '200k_iters',
    'lr4',
    'B2',
    'train_feat',
    'train_occ',
    'train_view',
    'pretrained_feat',
    'pretrained_view',
    'pretrained_occ',    
    'pretrained_munit',    
    'do_style_content',
    'reset_iter',
    'eval_boxes',    
    'train_munit',
    'smoothness_with_noloss',
    'no_bn',
    'fast_logging',
    'munit_loss_term_wt_0.1',
    'basic_view_loss',
    # 'style_view_loss',
    'cycle_style_view_loss',
    'simple_adaingen',
]


exps['trainer_rgb_no_bn_munit_simple_cross_0.1_dsn_frozen'] = [
    'clevr_sta', #mode
    'clevr_shapes_vqa_highres_singleobj', # dataset
    '200k_iters',
    'lr4',
    'B2',
    'train_feat',
    'train_occ',
    'train_view',
    'pretrained_feat',
    'pretrained_view',
    'pretrained_occ',    
    'pretrained_munit',    
    'do_style_content',
    'reset_iter',
    'eval_boxes',    
    'train_munit',
    'smoothness_with_noloss',
    'no_bn',
    'fast_logging',
    'frozen_feat',
    'munit_loss_term_wt_0.1',
    'basic_view_loss',
    # 'style_view_loss',
    'cycle_style_view_loss',
    'simple_adaingen',
]


exps['trainer_rgb_occ_no_bn'] = [
    'clevr_sta', # mode
    'clevr_veggies_sta_single_norotate', # dataset
    '200k_iters',
    'lr3',
    'B2',
    'train_feat',
    'train_occ', 
    'train_view', 
    'do_shape',
    'reset_iter',
    'replaceRD',
    'pretrained_feat',
    # 'pretrained_occ',      
    'no_bn',
    'fast_logging',
]


































############################################################################################ MULTIVIEW ##########################################################################################################################################

########## CLEVR specific stuff ends here ###############

groups['do_munit_vgg_perceptual'] = [
    'munit_vgg_perceptual_loss = True'
]

groups['no_feat_recons_munit'] = [
    'munit_recon_x_w = 0',
    'munit_recon_x_cyc_w = 0',
]

groups['munit_loss_term_wt_1'] = [
    'munit_loss_weight = 1'
]

groups['munit_loss_term_wt_0.1'] = [
    'munit_loss_weight = 0.1'
]

groups['munit_loss_term_wt_0.5'] = [
    'munit_loss_weight = 0.5'
]

groups['do_2d_munit'] = [
    'do_2d_style_munit = True',
    'do_3d_style_munit = False',
    'munit_input_dim_b = 3',
    'munit_input_dim_a = 3'            
]

groups['munit_loss_term_wt_0'] = [
    'munit_loss_weight = 0'
]

groups['munit_dynamic_layers1_adain01'] = [
    'decodersimple_3d_dynamic_num_conv_layers = 1',
    'use_dynamic_decoder_3d_simple = True',
    'adain_layers = [0,1]'
]

groups['munit_dynamic_layers2_adain03'] = [
    'decodersimple_3d_dynamic_num_conv_layers = 2',
    'use_dynamic_decoder_3d_simple = True',
    'adain_layers = [0,3]'
]

groups['munit_dynamic_layers2_adain12'] = [
    'decodersimple_3d_dynamic_num_conv_layers = 2',
    'use_dynamic_decoder_3d_simple = True',
    'adain_layers = [1,2]'
]

groups['quantize_object_no_detach_rotate_10'] = [
    'object_quantize = True',
    # 'object_quantize_init = "offline_obj_cluster/02_m144x144x144_1e-3_F32_Oc_c1_s1_aet_ns_trainer_occ_no_bn_onlyocc_cluster_centers_Example_50.npy"',
    # 'object_quantize_init = "offline_obj_cluster/02_m144x144x144_p128x128_1e-3_F32_Oc_c1_s1_V_d32_c1_aet_ns_trainer_rgb_occ_no_bn_replaceRD_cluster_centers_Example_50.npy"',
    # 'object_quantize_init = "offline_obj_cluster/02_m144x144x144_p128x128_1e-3_F32_Oc_c1_s1_V_d32_c1_aet_ns_trainer_rgb_occ_no_bn_cluster_centers_Example_50.npy"',
    'object_quantize_dictsize = 10',
    # 'object_quantize_init = "offline_obj_cluster/01_m144x144x144_p128x128_1e-3_F32_Oc_c1_s1_V_d32_c1_ae_contentt_ns_trainer_rgb_occ_no_bn_style_transfer_cluster_centers_Example_50.npy"',
    'object_quantize_comm_cost = 1.0',
    'detach_background = False',
    'vq_rotate = True',
]


groups['throw_thresh_0.975'] = [
    'throw_thresh = 0.975',
    'throw_away = True',
]

groups['throw_thresh_0.97'] = [
    'throw_thresh = 0.97',
    'throw_away = True',
]

groups['throw_thresh_0.98'] = [
    'throw_thresh = 0.98',
    'throw_away = True',
]


groups['quantize_object_no_detach_rotate_5_from_sup'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 5',
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    'vq_rotate = True',
    'from_sup = True',
    'object_quantize_sup_init = "dump/vqa_protos_shape_02_m144x144x144_1e-4_F32_rotMA500_0.25_0.75t_ns_clevr_multiple_trainer_hard_exp5_pret_moc_orient2_gt_occ_0.25_0.75_0.91.p"',
]

groups['quantize_object_no_detach_rotate_5_from_sup_occ'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 5',
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    'vq_rotate = True',
    'from_sup = True',
    'object_quantize_sup_init = "dump/vqa_protos_shape_02_m144x144x144_1e-3_F32_Oc_c1_s1_rotMA500_full_ns_trainer_occ_no_bn_0.97.p"',
]


groups['quantize_object_no_detach_rotate_5_supervised'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 5',
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    'vq_rotate = True',
    'use_supervised = True',
    'object_quantize_sup_init = "dump/vqa_protos_shape_02_m144x144x144_1e-4_F32_rotMA500_0.25_0.75t_ns_clevr_multiple_trainer_hard_exp5_pret_moc_orient2_gt_occ_0.25_0.75_0.91.p"',
]

groups['quantize_object_no_detach_rotate_5_from_sup_alien'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 5',
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    'vq_rotate = True',
    'from_sup = True',
    'object_quantize_sup_init = "dump/vqa_protos_shape_02_m144x144x144_1e-3_F32_Oc_c1_s1_rotMA500_full_ns_trainer_occ_no_bn_0.998.p"',
]

groups['quantize_object_no_detach_rotate_5_supervised_alien'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 5',
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    'vq_rotate = True',
    'use_supervised = True',
    'object_quantize_sup_init = "dump/vqa_protos_shape_02_m144x144x144_1e-3_F32_Oc_c1_s1_rotMA500_full_ns_trainer_occ_no_bn_0.998.p"',
]

groups['store_content_style_range'] = [
    'store_content_style_range = True'
]

groups['use_contrastive_examples'] = [
    'is_contrastive_examples = True',
    'load_content_style_range_from_file = True'
]

############## datasets ##############

# dims for mem
# SIZE = 32
import socket
if "Alien"  in socket.gethostname():
    SIZE = 24

else:
    SIZE = 36

# SIZE = 72

# 56




Z = SIZE*4
Y = SIZE*4
X = SIZE*4

Z2 = Z//2
Y2 = Y//2
X2 = X//2

BOX_SIZE = 16

K = 3 # how many objects to consider
S = 2
H = 256
W = 256
N = 3
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)

DATA_MOD = "aa"

# groups['clevr_veggies_sta_data'] = ['dataset_name = "clevr"',
#                              'H = %d' % H,
#                              'W = %d' % W,
#                              'trainset = "aat"',
#                              'dataset_list_dir = "/projects/katefgroup/datasets/clevr_veggies/npys"',
#                              'dataset_location = "/projects/katefgroup/datasets/clevr_veggies/npys"',
#                              'dataset_format = "npz"'
# ]
# Using darshan's code
groups['clevr_shapes_vqa_highres_multiobj'] = ['dataset_name = "clevr_vqa"',
                             'H = %d' % 320,
                             'W = %d' % 480,
                             'N = %d' % 10,
                             'PH = int(H/2.0)',
                             'PW = int(W/2.0)',                             
                             'root_keyword = "katefgroup"',
                             'trainset = "multi_obj_480_at"',
                             f'dataset_list_dir = "CHANGE_ME/clevr_vqa/raw/npys"',
                             f'dataset_location = "CHANGE_ME/clevr_vqa/raw/npys"',
                             f'root_dataset = "CHANGE_ME/clevr_vqa/raw/"',
                             'dataset_format = "npz"'
]
groups['clevr_shapes_vqa_highres_multiobj_test'] = ['dataset_name = "clevr_vqa"',
                             'H = %d' % 320,
                             'W = %d' % 480,
                             'N = %d' % 10,
                             'PH = int(H/2.0)',
                             'PW = int(W/2.0)',                             
                             'root_keyword = "katefgroup"',
                             'testset = "multi_obj_480_at"',
                             f'dataset_list_dir = "CHANGE_ME/clevr_vqa/raw/npys"',
                             f'dataset_location = "CHANGE_ME/clevr_vqa/raw/npys"',
                             f'root_dataset = "CHANGE_ME/clevr_vqa/raw/"',
                             'dataset_format = "npz"'
]

# Using darshan's code
groups['clevr_shapes_vqa_highres_singleobj'] = ['dataset_name = "clevr_vqa"',
                             'H = %d' % 320,
                             'W = %d' % 480,
                             'N = %d' % 10,
                             'PH = int(H/2.0)',
                             'PW = int(W/2.0)',                             
                             'root_keyword = "katefgroup"',
                             'trainset = "single_obj_large_480_gt"',
                             f'dataset_list_dir = "CHANGE_ME/clevr_vqa/raw/npys"',
                             f'dataset_location = "CHANGE_ME/clevr_vqa/raw/npys"',
                             f'root_dataset = "CHANGE_ME/clevr_vqa/raw/"',
                             'dataset_format = "npz"'
]

groups['clevr_shapes_vqa_highres_singleobj_test'] = ['dataset_name = "clevr_vqa"',
                             'H = %d' % 320,
                             'W = %d' % 480,
                             'N = %d' % 10,
                             'PH = int(H/2.0)',
                             'PW = int(W/2.0)',                             
                             'root_keyword = "katefgroup"',
                             'testset = "single_obj_large_480_gt"',
                             f'dataset_list_dir = "CHANGE_ME/clevr_vqa/raw/npys"',
                             f'dataset_location = "CHANGE_ME/clevr_vqa/raw/npys"',
                             f'root_dataset = "CHANGE_ME/clevr_vqa/raw/"',
                             'dataset_format = "npz"'
]

############## verify and execute ##############

def _verify_(s):
    varname, eq, val = s.split(' ')
    try:
        assert varname in globals()
        assert eq == '='
        assert type(s) is type('')
    except Exception as e:
        print(e)
        st()

print(current)
assert current in exps
for group in exps[current]:
    print("  " + group)
    if group not in groups:
      st()
      assert False
    for s in groups[group]:
        print("    " + s)
        _verify_(s)
        exec(s)

s = "mod = " + mod
_verify_(s)

exec(s)
import getpass
username = getpass.getuser()
import socket
hostname = socket.gethostname()

if 'compute' in hostname:
    if root_keyword == "katefgroup":
        root_location = "/projects/katefgroup/datasets/"
        dataset_list_dir = dataset_list_dir.replace("CHANGE_ME",root_location)
        dataset_location = dataset_location.replace("CHANGE_ME",root_location)
        root_dataset = root_dataset.replace("CHANGE_ME",root_location)
    elif root_keyword == "home":
        if 'shamit' in username:
            root_location = "/home/shamitl/datasets/"
        else:
            root_location = "/home/mprabhud/dataset/"
        dataset_list_dir = dataset_list_dir.replace("CHANGE_ME",root_location)
        dataset_location = dataset_location.replace("CHANGE_ME",root_location)
        root_dataset = root_dataset.replace("CHANGE_ME",root_location)
elif 'ip-' in hostname:
    root_location = "/projects"
    dataset_list_dir = dataset_list_dir.replace("CHANGE_ME",root_location)
    dataset_location = dataset_location.replace("CHANGE_ME",root_location)
    root_dataset = root_dataset.replace("CHANGE_ME",root_location)  
elif 'Alien' in hostname:
    root_location = "/media/mihir/dataset"
    dataset_list_dir = dataset_list_dir.replace("CHANGE_ME",root_location)
    dataset_location = dataset_location.replace("CHANGE_ME",root_location)
    root_dataset = root_dataset.replace("CHANGE_ME",root_location)