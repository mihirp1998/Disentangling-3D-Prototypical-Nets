import pretrained_nets_carla as pret_clevr
# import pretrained_nets_clevr as pret_clevr

exps = {}
groups = {}
group_parents = {}

############## preprocessing/shuffling ##############

############## modes ##############

groups['zoom'] = ['do_zoom = True']
groups['bigbird_sta'] = ['do_clevr_sta = True'] # Hack to run bigbird dataset with clevr code
groups['carla_det'] = ['do_clevr_sta = True'] # Hack to run carla dataset with clevr code
groups['replica_sta'] = ['do_clevr_sta = True'] # Hack to run carla dataset with clevr code

groups['carla_mot'] = ['do_carla_mot = True']
groups['carla_sta'] = ['do_carla_sta = True']
groups['carla_flo'] = ['do_carla_flo = True']
groups['clevr_sta'] = ['do_clevr_sta = True']
groups['style_sta'] = ['do_style_sta = True']
groups['nel_sta'] = ['do_nel_sta = True']
groups['style_sta'] = ['do_style_sta = True']
groups['carla_obj'] = ['do_carla_obj = True']
groups['mujoco_offline'] = ['do_mujoco_offline = True']

############## extras ##############
groups['rotate_combinations'] = ['gt_rotate_combinations = True']
groups['use_gt_centers'] = ['use_gt_centers = True']
groups['add_det_boxes'] = ['add_det_boxes = True']

groups['do_material'] = ['do_material = True','num_classes = 2','suffix = "material"','labels = ["rubber","metal"]']
groups['do_color'] = ['do_color = True','num_classes = 8','suffix = "color"','labels = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]']
groups['do_shape'] = ['do_shape = True','num_classes = 3','suffix = "shape"','labels = ["cube","sphere","cylinder"]']
groups['do_style'] = ['do_style = True']
groups['do_style_content'] = ['do_style_content = True']
groups['fixed_view'] = ['fixed_view = True']
groups['save_rgb'] = ['save_rgb = True']
groups['save_embed_tsne'] = ['save_embed_tsne = True']



groups['use_supervised'] = ['use_supervised = True']
groups['from_supervised'] = ['from_supervised = True']
groups['throw_away'] = ['']
groups['shape_aug'] = ['shape_aug = True']
groups['rotate_aug'] = ['rotate_aug = True']

groups['debug_add'] = ['debug_add = True']
groups['debug_match'] = ['debug_match = True']
groups['normalize_contrast'] = ['normalize_contrast = True']
groups['replace_sc'] = ['replace_sc = True']
groups['save_gt_occs'] = ['save_gt_occs = True']
groups['use_gt_occs'] = ['use_gt_occs = True']
groups['remove_air'] = ['remove_air = True','use_gt_occs = True']
groups['do_match_det'] = ['do_match_det = True']









groups['weight_decay_1'] = ['weight_decay = 1']
groups['weight_decay_0.1'] = ['weight_decay = 0.1']
groups['weight_decay_0.01'] = ['weight_decay = 0.01']
groups['weight_decay_0.001'] = ['weight_decay = 0.001']
groups['weight_decay_0.0001'] = ['weight_decay = 0.0001']
groups['weight_decay_0.00001'] = ['weight_decay = 0.00001']


groups['large_num_objs'] = ['min_obj_aug = 1','max_obj_aug = 7']
groups['smoothness_with_noloss'] = ['smoothness_with_noloss = True']

groups['med_num_objs'] = ['min_obj_aug = 1','max_obj_aug = 4']


groups['quantize_object_no_detach_rotate_instances_vsmall'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 41',    
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    'vq_rotate = True',
    'use_instances_variation = True',
    'var_coeff = 0.001'
]

groups['quantize_object_no_detach_rotate_instances_vvsmall'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 41',    
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    'vq_rotate = True',
    'use_instances_variation = True',
    'var_coeff = 0.0001'
]


groups['quantize_object_no_detach_rotate_instances_big'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 41',    
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    'vq_rotate = True',
    'use_instances_variation = True',
    'var_coeff = 0.1'
]


groups['quantize_object_no_detach_rotate_instances_vbig'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 41',    
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    'vq_rotate = True',
    'use_instances_variation = True',
    'var_coeff = 1.0'
]



groups['quantize_object_no_detach_rotate_instances_all_vbig'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 41',    
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    'vq_rotate = True',
    'use_instances_variation_all = True',
    'var_coeff = 1.0',    
]


groups['quantize_object_no_detach_rotate_instances_all_vsmall'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 41',    
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    'vq_rotate = True',
    'use_instances_variation_all = True',
    'var_coeff = 0.001',
]

groups['quantize_object_no_detach'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 41',
    'object_quantize_comm_cost = 0.25',
    # 'object_quantize_init = "offline_obj_cluster/cluster_centers_41.npy"',
    'detach_background = False',
]
groups['quantize_object_no_detach_ema'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 41',
    'object_quantize_comm_cost = 0.25',
    'object_quantize_init = "offline_obj_cluster/cluster_centers_41.npy"',
    'detach_background = False',
    'object_ema = True',    
]

groups['quantize_object_no_detach_no_cluster'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 41',
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    # 'object_quantize_init = "offline_obj_cluster/cluster_centers_41.npy"',
]

groups['quantize_object_no_detach_no_cluster_ema'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 41',
    'object_quantize_comm_cost = 0.25',
    'detach_background = False',
    'object_ema = True',

    # 'object_quantize_init = "offline_obj_cluster/cluster_centers_41.npy"',
]

groups['quantize_object_init_cluster'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 41',
    'object_quantize_comm_cost = 0.25',

]

groups['quantize_object_high_coef'] = [
    'object_quantize = True',
    'object_quantize_dictsize = 41',
    'object_quantize_comm_cost = 0.25',
    'object_quantize_init = "offline_obj_cluster/cluster_centers_41.npy"',
    'quantize_loss_coef = 5.0'
]

groups['train_feat_res'] = [
    'do_feat = True',
    'feat_dim = 32',
    'feat_do_rt = True',
    'feat_do_flip = True',
    'feat_do_resnet = True',
]
groups['train_feat_sb'] = [
    'do_feat = True',
    'feat_dim = 32',
    'feat_do_sb = True',
    'feat_do_resnet = True',
    'feat_do_flip = True',
    'feat_do_rt = True',
]
groups['train_occ_no_coeffs'] = [
    'do_occ = True',
    'occ_do_cheap = True',
]
groups['train_occ'] = [
    'do_occ = True',
    'occ_do_cheap = True',
    'occ_coeff = 1.0',
    'occ_smooth_coeff = 1.0',
]
groups['train_occ_less_smooth'] = [
    'do_occ = True',
    'occ_do_cheap = True',
    'occ_coeff = 1.0',
    'occ_smooth_coeff = 0.1',
]
groups['train_view'] = [
    'do_view = True',
    'view_depth = 32',
    'view_l1_coeff = 1.0',
]
groups['train_munit'] = [
    'do_munit = True'
]

groups['train_smoothnet'] = [
    'do_smoothnet = True'
]

groups['train_smoothnet1'] = [
    'do_smoothnet = True',
    'smoothness_recons_loss_weight = 0',
    'smoothness_gradient_loss_weight = 0.001',
]

groups['train_smoothnet2'] = [
    'do_smoothnet = True',
    'smoothness_recons_loss_weight = 0',
    'smoothness_gradient_loss_weight = 0.01',
]

groups['train_smoothnet3'] = [
    'do_smoothnet = True',
    'smoothness_recons_loss_weight = 0',
    'smoothness_gradient_loss_weight = 0.0001',
]


groups['style_view_loss'] = [
    'style_view_loss = True'
]


groups['cycle_style_view_loss'] = [
    'cycle_style_view_loss = True'
]


groups['simple_adaingen'] = [
    'simple_adaingen = True',
]



groups['do_munit_fewshot'] = [
    'do_munit_fewshot = True',
]


groups['run_few_shot_on_munit'] = [
    'run_few_shot_on_munit = True',
]

groups['avg_3d'] = [
    'avg_3d = True',
]


groups['basic_view_loss'] = [
    'basic_view_loss = True'
]



groups['train_render'] = [
    'do_render = True',
    'render_depth = 32',
    'render_l1_coeff = 1.0',
]
groups['train_view_accu_render'] = [
    'do_view = True',
    'view_depth = 32',
    'view_l1_coeff = 1.0',
    'view_accu_render = True',
]
groups['train_view_accu_render_unps_gt'] = [
    'do_view = True',
    'view_depth = 32',
    'view_l1_coeff = 1.0',
    'view_accu_render = True',
    'view_accu_render_unps = True',
    'view_accu_render_gt = True',
]
groups['train_view_accu_render_gt'] = [
    'do_view = True',
    'view_depth = 32',
    'view_l1_coeff = 1.0',
    'view_accu_render = True',
    'view_accu_render_gt = True',
]

groups['do_hard_eval'] = [
    'hard_eval = True',
]

groups['train_occ_notcheap'] = [
    'do_occ = True',
    'occ_coeff = 1.0',
    'occ_do_cheap = False',
    'occ_smooth_coeff = 0.1',
]
groups['train_emb2D'] = [
    'do_emb2D = True',
    'emb_2D_smooth_coeff = 0.01',
    'emb_2D_ml_coeff = 1.0',
    'emb_2D_l2_coeff = 0.1',
    'emb_2D_mindist = 32.0',
    'emb_2D_num_samples = 2',
]
groups['train_emb3D'] = [
    'do_emb3D = True',
    'emb_3D_smooth_coeff = 0.01',
    'emb_3D_ml_coeff = 1.0',
    'emb_3D_l2_coeff = 0.1',
    'emb_3D_mindist = 16.0',
    'emb_3D_num_samples = 2',
]
groups['train_emb3D_o'] = [
    'do_emb3D = True',
    'emb_3D_smooth_coeff = 0.01',
    'emb_3D_ml_coeff = 1.0',
    'emb_3D_l2_coeff = 0.1',
    'emb_3D_mindist = 16.0',
    'emb_3D_num_samples = 100',
    'emb3D_o = True',
    'do_eval_boxes = True',

]
groups['train_emb3D_moc'] = [
    'moc = True',
]
groups['eval_boxes'] = [
  'do_eval_boxes = True'
]

groups['empty_table'] = [
  'do_empty = True'
]

groups['eval_recall_summ_o'] = [
  'eval_recall_summ_o = True'
]


groups['filter_boxes'] = [
    'filter_boxes = True',
    'vq_rotate = True',    
]


groups['filter_boxes_cs'] = [
    'filter_boxes = True',
    'vq_rotate = True',
    'cs_filter = True',
]


groups['filter_boxes_100'] = [
    'filter_boxes = True',
    'vq_rotate = True',    
    'object_quantize_dictsize = 100',
]

groups['filter_boxes_50'] = [
    'filter_boxes = True',
    'vq_rotate = True',    
    'object_quantize_dictsize = 50',
]

groups['filter_boxes_cs_100'] = [
    'filter_boxes = True',
    'vq_rotate = True',
    'object_quantize_dictsize = 100',    
    'cs_filter = True',
]


groups['learn_supervised_embeddings'] = [
    'learn_linear_embeddings = True',
    'supervised_embedding_loss_coeff = 1.0',
]





############## net configs ##############
groups['new_distance_thresh'] = ['dict_distance_thresh = 750']


groups['train_det_px'] = [
    'do_pixor_det = True',    
]

groups['train_det_px_calc_mean'] = [
    'do_pixor_det = True',
    'calculate_mean = True',
]


groups['train_det_px_calc_std'] = [
    'do_pixor_det = True',    
    'calculate_std = True',
]


groups['train_det'] = [
    'do_det = True',    
]

groups['train_det_deep'] = [
    'do_det = True',
    'deeper_det = True'
]

groups['aug_object_ent'] = [
    'aug_object_ent = True',
    'aug_det = True',
    'store_ent_obj = True',
]


groups['aug_object_ent_dis'] = [
    'aug_object_ent_dis = True',
    'aug_det = True',
    'store_ent_dis_obj = True',
   'simple_adaingen = True',    
]



groups['add_random_noise'] = [
    'add_random_noise = True',
]

groups['aug_object_dis'] = [
    'aug_object_dis = True',
    'aug_det = True',
    'store_dis_obj = True',
    'simple_adaingen = True',
]


groups['accumulate_grad'] = [
    'accumulate_grad = True',
    'accumulation_steps = 3',
]


groups['do_munit_det'] = [
    'do_munit_det = True',
]
groups['og_debug'] = [
    'og_debug = True',
]

groups['single_view'] = [
    'single_view = True',
]

groups['debug_aug'] = ['debug_aug = True']
groups['store_ent_obj'] = ['store_ent_obj = True','store_obj = True']
groups['store_dis_obj'] = ['store_dis_obj = True','store_obj = True']

groups['do_material_content'] = ['do_material_content = True']
groups['do_color_content'] = ['do_color_content = True']

groups['train_det_gt_px'] = [
    'do_gt_pixor_det = True',    
]

groups['online_cluster_20']= [
    'online_cluster = True',
    'object_quantize_dictsize = 20',
    'cluster_iters = 3000',
    'online_cluster_eval = True',    
    'initial_cluster_size = 20000',

]

groups['do_moc']= [
    'moc = True',
    'moc_qsize = 100000',
]

groups['offline_cluster_100']= [
    'offline_cluster = True',
    'offline_cluster_pool_size = 120',
    'object_quantize_dictsize = 100',
]


groups['offline_cluster_eval_kmeans'] = [
    'offline_cluster_eval = True',
    'offline_cluster_eval_iters = 12000',
    'use_kmeans = True',
]

groups['offline_cluster_eval_vqvae'] = [
    'offline_cluster_eval = True',
    'offline_cluster_eval_iters = 12000',
    'use_vq_vae = True',    
]



groups['offline_cluster_eval_vqvae_rotate'] = [
    'offline_cluster_eval = True',
    'offline_cluster_eval_iters = 12000',
    'use_vq_vae = True',    
    'vq_rotate = True',
]


groups['offline_cluster_eval_vqvae_rotate_instances'] = [
    'offline_cluster_eval = True',
    'offline_cluster_eval_iters = 12000',
    'use_vq_vae = True',    
    'vq_rotate = True',
    'use_instances_variation = True',
    'var_coeff = 0.1',
]

groups['offline_cluster_eval_vqvae_rotate_instances_all_vbig'] = [
    'offline_cluster_eval = True',
    'offline_cluster_eval_iters = 12000',
    'use_vq_vae = True',    
    'vq_rotate = True',
    'use_instances_variation_all = True',
    'var_coeff = 1.0',    
    'num_rand_samps = 30',

]

groups['offline_cluster_eval_vqvae_rotate_instances_all_vsmall'] = [
    'offline_cluster_eval = True',
    'offline_cluster_eval_iters = 12000',
    'use_vq_vae = True',    
    'vq_rotate = True',
    'use_instances_variation_all = True',
    'var_coeff = 0.001',
]


groups['offline_cluster_eval_vqvae_rotate_instances_all_vvsmall'] = [
    'offline_cluster_eval = True',
    'offline_cluster_eval_iters = 12000',
    'use_vq_vae = True',    
    'vq_rotate = True',
    'use_instances_variation_all = True',
    'var_coeff = 0.0001',
]


groups['low_dict_size']= [
    'low_dict_size = True',
]
groups['hard_vis']= [
    'hard_vis = True',
]

groups['do_moc2d']= [
    'moc_2d = True',
]
groups['reset_iter'] = [
    'reset_iter = True',
]

groups['imgnet'] = [
    'imgnet = True',

]
groups['train_preocc'] = ['do_preocc = True']
groups['no_bn'] = ['no_bn = True']

groups['do_gen_pcds'] = [
  'GENERATE_PCD_BBOX = True'
]
groups['object_specific'] = [
    'do_object_specific = True',
    'do_eval_boxes = True',
]
groups['debug'] = [
    'do_debug = True',
    'moc_qsize = 1000',
    'offline_cluster_pool_size = 50',    
    'offline_cluster_eval_iters = 101',
    'eval_compute_freq = 1',    
    'log_freq_train = 1',
    'log_freq_val = 1',
    'log_freq_test = 1',
    'log_freq = 1',
]
groups['train_feat'] = [
    'do_feat = True',
    'feat_dim = 32',
]

groups['style_transfer'] = [
    'style_transfer = True',
    'do_eval_boxes = True',
]



groups['quantize_vox_512'] = [
    'voxel_quantize = True',
    'voxel_quantize_dictsize = 512',
    'voxel_quantize_comm_cost = 0.25',
    # 'voxel_quantize_init = 0.25',
]

groups['quantize_vox_256'] = [
    'voxel_quantize = True',
    'voxel_quantize_dictsize = 256',
    'voxel_quantize_comm_cost = 0.25',
    # 'voxel_quantize_init = 0.25',
]

groups['quantize_vox_128'] = [
    'voxel_quantize = True',
    'voxel_quantize_dictsize = 128',
    'voxel_quantize_comm_cost = 0.25',
    # 'voxel_quantize_init = 0.25',
]
groups['quantize_vox_64'] = [
    'voxel_quantize = True',
    'voxel_quantize_dictsize = 64',
    'voxel_quantize_comm_cost = 0.25',
    # 'voxel_quantize_init = 0.25',
]

groups['quantize_vox_32'] = [
    'voxel_quantize = True',
    'voxel_quantize_dictsize = 32',
    'voxel_quantize_comm_cost = 0.25',
    # 'voxel_quantize_init = 0.25',
]

# Store prototypes from the input entities. Use them to initialize protos in future exps
groups['set_create_protos'] = [
    'create_prototypes = True' 
]

groups['normalize_style'] = [
    'normalize_style = True' 
]

groups['quantize_vox_1024'] = [
    'voxel_quantize = True',
    'voxel_quantize_dictsize = 512',
    'voxel_quantize_comm_cost = 0.25',
    # 'voxel_quantize_init = 0.25',
]






groups['eval_recall'] = ['do_eval_recall = True']

groups['eval_recall_o'] = [
    'do_eval_recall = True',
    'eval_recall_o = True',
    'do_eval_boxes = True',
    'eval_recall_log_freq = 1',
    # make sure it is a multiple of eval_recall_log_freq
    'eval_compute_freq = 500',
]
groups['eval_recall_o_vbig_pool'] = [
    'do_eval_recall = True',
    'eval_recall_o = True',
    'do_eval_boxes = True',
    'eval_recall_log_freq = 1',
    # make sure it is a multiple of eval_recall_log_freq
    'pool_size = 5000',
    'eval_compute_freq = 500',
]

groups['eval_recall_o_slow'] = [
    'do_eval_recall = True',
    'eval_recall_o = True',
    'do_eval_boxes = True',
    'eval_recall_log_freq = 1',
    # make sure it is a multiple of eval_recall_log_freq
    'eval_compute_freq = 1000',
]

groups['eval_recall_o_quicker_small_pool'] = [
    'do_eval_recall = True',
    'eval_recall_o = True',
    'do_eval_boxes = True',
    'eval_recall_log_freq = 1',
    'pool_size = 100',
    # make sure it is a multiple of eval_recall_log_freq
    'eval_compute_freq = 1',
]
groups['eval_recall_o_quicker_big_pool'] = [
    'do_eval_recall = True',
    'eval_recall_o = True',
    'do_eval_boxes = True',
    'eval_recall_log_freq = 1',
    'pool_size = 1000',    
    # make sure it is a multiple of eval_recall_log_freq
    'eval_compute_freq = 1',
]
groups['eval_recall_o_quicker_big_pool1'] = [
    'do_eval_recall = True',
    'eval_recall_o = True',
    'do_eval_boxes = True',
    'eval_recall_log_freq = 1',
    'pool_size = 500',    
    # make sure it is a multiple of eval_recall_log_freq
    'eval_compute_freq = 1',
]
groups['eval_recall_o_quicker_big_pool2'] = [
    'do_eval_recall = True',
    'eval_recall_o = True',
    'do_eval_boxes = True',
    'eval_recall_log_freq = 1',
    'pool_size = 250',    
    # make sure it is a multiple of eval_recall_log_freq
    'eval_compute_freq = 1',
]



groups['eval_recall_o_quicker_vbig_pool'] = [
    'do_eval_recall = True',
    'eval_recall_o = True',
    'do_eval_boxes = True',
    'eval_recall_log_freq = 1',
    'pool_size = 5000',
    # make sure it is a multiple of eval_recall_log_freq
    'eval_compute_freq = 1',
]
groups['debug_eval_recall_o'] = [
    'debug_eval_recall_o = True',
]

groups['orient_tensors_in_eval_recall_o'] = [
    'do_orientation = True',
]
groups['no_eval_recall'] = ['do_eval_recall = False']



groups['vis_clusters'] = ['vis_clusters = True']

groups['randomly_select_views'] = ['randomly_select_views = True']

groups['obj_multiview'] = ['obj_multiview = True']
groups['S3'] = ['S = 3']


groups['decay_lr'] = ['do_decay_lr = True']
groups['clip_grad'] = ['do_clip_grad = True']

groups['quick_snap'] = ['snap_freq = 1000','delete_checkpoints_older_than = 5']


groups['halucinate_vals_100'] = ['halucinate_vals = 100']
groups['halucinate_vals_50'] = ['halucinate_vals = 50']
groups['halucinate_vals_10'] = ['halucinate_vals = 4']

groups['quicker_snap'] = ['snap_freq = 50']
groups['quickest_snap'] = ['snap_freq = 5']
groups['superquick_snap'] = ['snap_freq = 1']

groups['use_det_boxes'] = ['use_det_boxes = True']
groups['summ_all'] = ['summ_all = True']
groups['onlyocc'] = ['onlyocc = True']
groups['replaceRD'] = ['replaceRD = True']

# groups['use_supervised'] = ['use_supervised = True']




groups['create_example_dict_100'] = ['create_example_dict = True','object_quantize_dictsize = 100']
groups['create_example_dict_82'] = ['create_example_dict = True','object_quantize_dictsize = 82']
groups['create_example_dict_70'] = ['create_example_dict = True','object_quantize_dictsize = 70']
groups['create_example_dict_52'] = ['create_example_dict = True','object_quantize_dictsize = 52']
groups['create_example_dict_50'] = ['create_example_dict = True','object_quantize_dictsize = 50']
groups['create_example_dict_3'] = ['create_example_dict = True','object_quantize_dictsize = 3']
groups['create_example_dict_5'] = ['create_example_dict = True','object_quantize_dictsize = 5']
groups['create_example_dict_10'] = ['create_example_dict = True','object_quantize_dictsize = 10']
groups['create_example_dict_25'] = ['create_example_dict = True','object_quantize_dictsize = 25']
groups['create_example_dict_40'] = ['create_example_dict = True','object_quantize_dictsize = 40']
groups['only_embed'] = ['only_embed = True']


groups['profile_time'] = ['profile_time = True']
groups['low_res'] = ['low_res = True']
groups['cpu'] = ['cpu = True']

groups['eval_quantize'] = ['eval_quantize = True']

groups['use_2d_boxes'] = ['use_2d_boxes = True']


groups['no_shuf'] = ['shuffle_train = False',
                     'shuffle_val = False',
                     'shuffle_test = False',
]


groups['no_shuf_val'] = [
                     'shuffle_val = False',
]

groups['shuf'] = ['shuffle_train = True',
                  'shuffle_val = True',
                    'shuffle_test = True',
]

groups['no_backprop'] = ['backprop_on_train = False',
                         'backprop_on_val = False',
                         'backprop_on_test = False',
]
groups['gt_ego'] = ['ego_use_gt = True']
groups['precomputed_ego'] = ['ego_use_precomputed = True']
groups['aug3D'] = ['do_aug3D = True']
groups['aug2D'] = ['do_aug2D = True']

groups['sparsify_pointcloud_10k'] = ['do_sparsify_pointcloud = 10000']
groups['sparsify_pointcloud_1k'] = ['do_sparsify_pointcloud = 1000']

groups['horz_flip'] = ['do_horz_flip = True']
groups['synth_rt'] = ['do_synth_rt = True']
groups['piecewise_rt'] = ['do_piecewise_rt = True']
groups['synth_nomotion'] = ['do_synth_nomotion = True']
groups['aug_color'] = ['do_aug_color = True']
groups['break_constraint'] = ['break_constraint = True']


# groups['eval'] = ['do_eval = True']
groups['random_noise'] = ['random_noise = True']
groups['eval_map'] = ['do_eval_map = True']
groups['save_embs'] = ['do_save_embs = True']
groups['save_ego'] = ['do_save_ego = True']
groups['save_vis'] = ['do_save_vis = True']

groups['profile'] = ['do_profile = True',
                     'log_freq_train = 100000000',
                     'log_freq_val = 100000000',
                     'log_freq_test = 100000000',
                     'max_iters = 20']

groups['B1'] = ['B = 1']
groups['B2'] = ['B = 2']
groups['B4'] = ['B = 4']
groups['B8'] = ['B = 8']
groups['B10'] = ['B = 10']
groups['B16'] = ['B = 16']
groups['B32'] = ['B = 32']
groups['B64'] = ['B = 64']
groups['B128'] = ['B = 128']
groups['lr0'] = ['lr = 0.0']
groups['lr2'] = ['lr = 1e-2']
groups['lr3'] = ['lr = 1e-3']
groups['2lr4'] = ['lr = 2e-4']
groups['5lr4'] = ['lr = 5e-4']
groups['lr4'] = ['lr = 1e-4']
groups['lr5'] = ['lr = 1e-5']
groups['lr6'] = ['lr = 1e-6']
groups['lr7'] = ['lr = 1e-7']
groups['lr8'] = ['lr = 1e-8']
groups['lr9'] = ['lr = 1e-9']
groups['lr12'] = ['lr = 1e-12']
groups['1_iters'] = ['max_iters = 1']
groups['3_iters'] = ['max_iters = 3']
groups['5_iters'] = ['max_iters = 5']
groups['6_iters'] = ['max_iters = 6']
groups['9_iters'] = ['max_iters = 9']
groups['21_iters'] = ['max_iters = 21']
groups['10_iters'] = ['max_iters = 10']
groups['20_iters'] = ['max_iters = 20']
groups['25_iters'] = ['max_iters = 25']
groups['30_iters'] = ['max_iters = 30']
groups['50_iters'] = ['max_iters = 50']
groups['100_iters'] = ['max_iters = 100']
groups['150_iters'] = ['max_iters = 150']
groups['200_iters'] = ['max_iters = 200']
groups['250_iters'] = ['max_iters = 250']
groups['300_iters'] = ['max_iters = 300']
groups['397_iters'] = ['max_iters = 397']
groups['400_iters'] = ['max_iters = 400']
groups['447_iters'] = ['max_iters = 447']
groups['500_iters'] = ['max_iters = 500']
groups['850_iters'] = ['max_iters = 850']
groups['1000_iters'] = ['max_iters = 1000']
groups['2000_iters'] = ['max_iters = 2000']
groups['2445_iters'] = ['max_iters = 2445']
groups['3000_iters'] = ['max_iters = 3000']
groups['4000_iters'] = ['max_iters = 4000']
groups['4433_iters'] = ['max_iters = 4433']
groups['5000_iters'] = ['max_iters = 5000']
groups['10000_iters'] = ['max_iters = 10000']
groups['1k_iters'] = ['max_iters = 1000']
groups['2k_iters'] = ['max_iters = 2000']
groups['5k_iters'] = ['max_iters = 5000']
groups['10k_iters'] = ['max_iters = 10000']
groups['20k_iters'] = ['max_iters = 20000']
groups['30k_iters'] = ['max_iters = 30000']
groups['40k_iters'] = ['max_iters = 40000']
groups['50k_iters'] = ['max_iters = 50000']
groups['60k_iters'] = ['max_iters = 60000']
groups['80k_iters'] = ['max_iters = 80000']
groups['100k_iters'] = ['max_iters = 100000']
groups['100k10_iters'] = ['max_iters = 100010']
groups['200k_iters'] = ['max_iters = 200000']
groups['300k_iters'] = ['max_iters = 300000']
groups['400k_iters'] = ['max_iters = 400000']
groups['500k_iters'] = ['max_iters = 500000']

groups['resume'] = ['do_resume = True']
# groups['total_init'] = ['total_init = pret_carl.total_init']
groups['reset_iter'] = ['reset_iter = True']

groups['fastest_logging'] = ['log_freq_train = 1',
                             'log_freq_val = 1',
                             'log_freq_test = 1',
                             'log_freq = 1']

groups['fastest1_logging'] = ['log_freq_train = 10',
                             'log_freq_val = 10',
                             'log_freq_test = 10',
                             'log_freq = 10']

groups['faster_logging'] = ['log_freq_train = 50',
                            'log_freq_val = 50',
                            'log_freq_test = 50',
                            'log_freq = 50',
]
groups['fast_logging'] = ['log_freq_train = 250',
                          'log_freq_val = 250',
                          'log_freq_test = 250',
                          'log_freq = 250',
]
groups['slow_logging'] = ['log_freq_train = 500',
                          'log_freq_val = 500',
                          'log_freq_test = 500',
                          'log_freq = 500',                          
]
groups['slower_logging'] = ['log_freq_train = 1000',
                            'log_freq_val = 1000',
                            'log_freq_test = 1000',
                            'log_freq = 1000',                          

]
groups['no_logging'] = ['log_freq_train = 100000000000',
                        'log_freq_val = 100000000000',
                        'log_freq_test = 100000000000',
                        'log_freq = 100000000000',                        
]



groups['fastest_logging_group'] = ['log_freq = 1',

]
groups['fastest2_logging_group'] = ['log_freq = 20',
]
groups['faster_logging_group'] = ['log_freq = 50',
]
groups['fast_logging_group'] = ['log_freq = 100',
]
groups['slow_logging_group'] = ['log_freq = 500',                          
]
groups['slower_logging_group'] = ['log_freq = 1000',
]
groups['no_logging_group'] = ['log_freq = 100000000000',                        
]
groups['style_baseline'] = ['style_baseline = True',                        
]
# ############## pretrained nets ##############
groups['pretrained_feat'] = ['do_feat = True',
                             'feat_init = "' + pret_clevr.feat_init + '"',
                             # 'feat_do_vae = ' + str(pret_clevr.feat_do_vae),
                             # 'feat_dim = %d' % pret_clevr.feat_dim,
]

groups['pretrained_munit'] = [
                             'munit_init = "' + pret_clevr.munit_init + '"',
                             # 'feat_do_vae = ' + str(pret_clevr.feat_do_vae),
                             # 'feat_dim = %d' % pret_clevr.feat_dim,
]

groups['pretrained_smoothnet'] = [
                             'smoothnet_init = "' + pret_clevr.smoothnet_init + '"',
                             # 'feat_do_vae = ' + str(pret_clevr.feat_do_vae),
                             # 'feat_dim = %d' % pret_clevr.feat_dim,
]

groups['pretrained_view'] = ['do_view = True',
                             'view_init = "' + pret_clevr.view_init + '"',
                             # 'view_depth = %d' %  pret_clevr.view_depth,
                             # 'view_use_halftanh = ' + str(pret_clevr.view_use_halftanh),
                             # 'view_pred_embs = ' + str(pret_clevr.view_pred_embs),
                             # 'view_pred_rgb = ' + str(pret_clevr.view_pred_rgb),
]
groups['pretrained_det'] = ['det_init = "' + pret_clevr.det_init + '"',
                             # 'view_depth = %d' %  pret_clevr.view_depth,
                             # 'view_use_halftanh = ' + str(pret_clevr.view_use_halftanh),
                             # 'view_pred_embs = ' + str(pret_clevr.view_pred_embs),
                             # 'view_pred_rgb = ' + str(pret_clevr.view_pred_rgb),
]

groups['pretrained_quantized'] = ['quant_init = "' + pret_clevr.quant_init + '"',
]


groups['pretrained_pixor'] = ['pixor_init = "' + pret_clevr.pixor_init + '"',
]

groups['pretrained_flow'] = ['do_flow = True',
                             'flow_init = "' + pret_clevr.flow_init + '"',
]
groups['pretrained_tow'] = ['do_tow = True',
                            'tow_init = "' + pret_clevr.tow_init + '"',
]
groups['pretrained_emb2D'] = ['do_emb2D = True',
                              'emb2D_init = "' + pret_clevr.emb2D_init + '"',
                              # 'emb_dim = %d' % pret_clevr.emb_dim,
]
groups['pretrained_occ'] = [
                            'occ_init = "' + pret_clevr.occ_init + '"',
                            # 'occ_do_cheap = ' + str(pret_clevr.occ_do_cheap),
]
groups['pretrained_preocc'] = [
    'do_preocc = True',
    'preocc_init = "' + pret_clevr.preocc_init + '"',
]
groups['pretrained_vis'] = ['do_vis = True',
                            'vis_init = "' + pret_clevr.vis_init + '"',
                            # 'occ_cheap = ' + str(pret_clevr.occ_cheap),
]

groups['only_cs_vis'] = ['only_cs_vis = True','cs_filter = True']
groups['replace_with_cs'] = ['replace_with_cs = True']

groups['only_q_vis'] = ['only_q_vis = True','cs_filter = False']
groups['q_cs_vis'] = ['only_q_vis = True','cs_filter = True']



groups['self_improve_once'] = ['self_improve_once = True']
groups['self_improve_once_maskout'] = ['self_improve_once = True','maskout = True']

groups['maskout'] = ['maskout = True']

groups['high_neg'] = ['alpha_pos = 1.0','beta_neg = 2.0']


groups['fast_orient'] = ['fast_orient = True']

groups['frozen_smoothnet'] = ['do_freeze_smoothnet = True']
groups['frozen_munit'] = ['do_freeze_munit = True']
groups['frozen_feat'] = ['do_freeze_feat = True', 'do_feat = True']
groups['frozen_view'] = ['do_freeze_view = True', 'do_view = True']
groups['frozen_vis'] = ['do_freeze_vis = True', 'do_vis = True']
groups['frozen_flow'] = ['do_freeze_flow = True', 'do_flow = True']
groups['frozen_emb2D'] = ['do_freeze_emb2D = True', 'do_emb2D = True']
groups['frozen_occ'] = ['do_freeze_occ = True', 'do_occ = True']
# groups['frozen_ego'] = ['do_freeze_ego = True', 'do_ego = True']
# groups['frozen_inp'] = ['do_freeze_inp = True', 'do_inp = True']
