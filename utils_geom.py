import torch
import utils_basic
import numpy as np
import ipdb
st = ipdb.set_trace

def eye_3x3(B):
    rt = torch.eye(3, device=torch.device('cuda')).view(1,3,3).repeat([B, 1, 1])
    return rt

def eye_3x3s(B, S):
    rt = torch.eye(3, device=torch.device('cuda')).view(1,1,3,3).repeat([B, S, 1, 1])
    return rt

def eye_4x4(B):
    rt = torch.eye(4, device=torch.device('cuda')).view(1,4,4).repeat([B, 1, 1])
    return rt

def eye_4x4s(B, S):
    rt = torch.eye(4, device=torch.device('cuda')).view(1,1,4,4).repeat([B, S, 1, 1])
    return rt

def merge_rt(r, t):
    # r is B x 3 x 3
    # t is B x 3
    B, C, D = list(r.shape)
    B2, D2 = list(t.shape)
    assert(C==3)
    assert(D==3)
    assert(B==B2)
    assert(D2==3)
    t = t.view(B, 3)
    rt = eye_4x4(B)
    rt[:,:3,:3] = r
    rt[:,:3,3] = t
    return rt

def deg2rad(deg):
    return deg/180.0*np.pi

def in_un(content_feat):
    # assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    # style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat



def adin(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 5)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1,1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1,1)
    return feat_mean, feat_std

    
def apply_4x4(RT, xyz):
    B, N, _ = list(xyz.shape)
    ones = torch.ones_like(xyz[:,:,0:1])
    xyz1 = torch.cat([xyz, ones], 2)
    xyz1_t = torch.transpose(xyz1, 1, 2)
    # this is B x 4 x N
    xyz2_t = torch.matmul(RT, xyz1_t)
    xyz2 = torch.transpose(xyz2_t, 1, 2)
    xyz2 = xyz2[:,:,:3]
    return xyz2

def split_rt_single(rt):
    r = rt[:3, :3]
    t = rt[:3, 3].view(3)
    return r, t

def split_rt(rt):
    r = rt[:, :3, :3]
    t = rt[:, :3, 3].view(-1, 3)
    return r, t

def safe_inverse_single(a):
    r, t = split_rt_single(a)
    t = t.view(3,1)
    r_transpose = r.t()
    inv = torch.cat([r_transpose, -torch.matmul(r_transpose, t)], 1)
    bottom_row = a[3:4, :] # this is [0, 0, 0, 1]
    # bottom_row = torch.tensor([0.,0.,0.,1.]).view(1,4) 
    inv = torch.cat([inv, bottom_row], 0)
    return inv

# def safe_inverse(a):
#     B, _, _ = list(a.shape)
#     inv = torch.zeros(B, 4, 4).cuda()
#     for b in range(B):
#         inv[b] = safe_inverse_single(a[b])
#     return inv

def safe_inverse(a): #parallel version
    B, _, _ = list(a.shape)
    inv = a.clone()
    r_transpose = a[:, :3, :3].transpose(1,2) #inverse of rotation matrix

    inv[:, :3, :3] = r_transpose
    inv[:, :3, 3:4] = -torch.matmul(r_transpose, a[:, :3, 3:4])

    return inv

def get_camM_T_camXs(origin_T_camXs, ind=0):
    B, S = list(origin_T_camXs.shape)[0:2]
    camM_T_camXs = torch.zeros_like(origin_T_camXs)
    for b in range(B):
        camM_T_origin = safe_inverse_single(origin_T_camXs[b,ind])
        for s in range(S):
            camM_T_camXs[b,s] = torch.matmul(camM_T_origin, origin_T_camXs[b,s])
    return camM_T_camXs

def scale_intrinsics(K, sx, sy):
    fx, fy, x0, y0 = split_intrinsics(K)
    fx = fx*sx
    fy = fy*sy
    x0 = x0*sx
    y0 = y0*sy
    K = pack_intrinsics(fx, fy, x0, y0)
    return K

def split_intrinsics(K):
    # K is B x 3 x 3 or B x 4 x 4
    fx = K[:,0,0]
    fy = K[:,1,1]
    x0 = K[:,0,2]
    y0 = K[:,1,2]
    return fx, fy, x0, y0

def pack_intrinsics(fx, fy, x0, y0):
    B = list(fx.shape)[0]
    K = torch.zeros(B, 4, 4, dtype=torch.float32, device=torch.device('cuda'))
    K[:,0,0] = fx
    K[:,1,1] = fy
    K[:,0,2] = x0
    K[:,1,2] = y0
    K[:,2,2] = 1.0
    K[:,3,3] = 1.0
    return K

def depth2pointcloud(z, pix_T_cam):
    B, C, H, W = list(z.shape)
    y, x = utils_basic.meshgrid2D(B, H, W)
    z = torch.reshape(z, [B, H, W])
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    xyz = Pixels2Camera(x, y, z, fx, fy, x0, y0)
    return xyz

def depth2pointcloud_cpu(z, pix_T_cam):
    B, C, H, W = list(z.shape)
    y, x = utils_basic.meshgrid2D_cpu(B, H, W)
    z = torch.reshape(z, [B, H, W])
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    xyz = Pixels2Camera(x, y, z, fx, fy, x0, y0)
    return xyz


def Pixels2Camera(x,y,z,fx,fy,x0,y0):
    # x and y are locations in pixel coordinates, z is a depth image in meters
    # their shapes are B x H x W
    # fx, fy, x0, y0 are scalar camera intrinsics
    # returns xyz, sized [B,H*W,3]
    
    B, H, W = list(z.shape)

    fx = torch.reshape(fx, [B,1,1])
    fy = torch.reshape(fy, [B,1,1])
    x0 = torch.reshape(x0, [B,1,1])
    y0 = torch.reshape(y0, [B,1,1])
    
    # unproject
    EPS = 1e-6
    x = ((z+EPS)/fx)*(x-x0)
    y = ((z+EPS)/fy)*(y-y0)
    
    x = torch.reshape(x, [B,-1])
    y = torch.reshape(y, [B,-1])
    z = torch.reshape(z, [B,-1])
    xyz = torch.stack([x,y,z], dim=2)
    return xyz

def eul2rotm(rx, ry, rz):
    # inputs are shaped B
    # this func is copied from matlab
    # R = [  cy*cz   sy*sx*cz-sz*cx    sy*cx*cz+sz*sx
    #        cy*sz   sy*sx*sz+cz*cx    sy*cx*sz-cz*sx
    #        -sy            cy*sx             cy*cx]
    rx = torch.unsqueeze(rx, dim=1)
    ry = torch.unsqueeze(ry, dim=1)
    rz = torch.unsqueeze(rz, dim=1)
    # these are B x 1
    sinz = torch.sin(rz)
    siny = torch.sin(ry)
    sinx = torch.sin(rx)
    cosz = torch.cos(rz)
    cosy = torch.cos(ry)
    cosx = torch.cos(rx)
    r11 = cosy*cosz
    r12 = sinx*siny*cosz - cosx*sinz
    r13 = cosx*siny*cosz + sinx*sinz
    r21 = cosy*sinz
    r22 = sinx*siny*sinz + cosx*cosz
    r23 = cosx*siny*sinz - sinx*cosz
    r31 = -siny
    r32 = sinx*cosy
    r33 = cosx*cosy
    r1 = torch.stack([r11,r12,r13],dim=2)
    r2 = torch.stack([r21,r22,r23],dim=2)
    r3 = torch.stack([r31,r32,r33],dim=2)
    r = torch.cat([r1,r2,r3],dim=1)
    return r

def rotm2eul(r):
    # r is Bx3x3
    r00 = r[:,0,0]
    r10 = r[:,1,0]
    r11 = r[:,1,1]
    r12 = r[:,1,2]
    r20 = r[:,2,0]
    r21 = r[:,2,1]
    r22 = r[:,2,2]
    
    ## python guide:
    # if sy > 1e-6: # singular
    #     x = math.atan2(R[2,1] , R[2,2])
    #     y = math.atan2(-R[2,0], sy)
    #     z = math.atan2(R[1,0], R[0,0])
    # else:
    #     x = math.atan2(-R[1,2], R[1,1])
    #     y = math.atan2(-R[2,0], sy)
    #     z = 0
    
    sy = torch.sqrt(r00*r00 + r10*r10)
    
    cond = (sy > 1e-6)
    rx = torch.where(cond, torch.atan2(r21, r22), torch.atan2(-r12, r11))
    ry = torch.where(cond, torch.atan2(-r20, sy), torch.atan2(-r20, sy))
    rz = torch.where(cond, torch.atan2(r10, r00), torch.zeros_like(r20))

    # rx = torch.atan2(r21, r22)
    # ry = torch.atan2(-r20, sy)
    # rz = torch.atan2(r10, r00)
    # rx[cond] = torch.atan2(-r12, r11)
    # ry[cond] = torch.atan2(-r20, sy)
    # rz[cond] = 0.0
    return rx, ry, rz

def inflate_to_axis_aligned_boxlist(boxlist):
    B, N, D = list(boxlist.shape)
    assert(D==9)

    corners = transform_boxes_to_corners(boxlist) # corners is B x N x 8 x 3
    corners_max = torch.max(corners, dim=2)[0]
    corners_min = torch.min(corners, dim=2)[0]

    centers = (corners_max + corners_min)/2.0
    sizes = corners_max - corners_min
    rots = torch.zeros_like(sizes)

    # xc, yc, zc, lx, ly, lz, rx, ry, rz
    boxlist_norot = torch.cat([centers, sizes, rots], dim=2)
    # boxlist_norot is B x N x 9

    return boxlist_norot

def get_random_rt(B,
                  r_amount=5.0,
                  t_amount=1.0,
                  sometimes_zero=False):
    # t_amount is in meters
    # r_amount is in degrees
    
    r_amount = np.pi/180.0*r_amount

    ## translation
    tx = np.random.uniform(-t_amount, t_amount, size=B).astype(np.float32)
    ty = np.random.uniform(-t_amount/2.0, t_amount/2.0, size=B).astype(np.float32)
    tz = np.random.uniform(-t_amount, t_amount, size=B).astype(np.float32)
    
    ## rotation
    rx = np.random.uniform(-r_amount/2.0, r_amount/2.0, size=B).astype(np.float32)
    ry = np.random.uniform(-r_amount, r_amount, size=B).astype(np.float32)
    rz = np.random.uniform(-r_amount/2.0, r_amount/2.0, size=B).astype(np.float32)

    if sometimes_zero:
        rand = np.random.uniform(0.0, 1.0, size=B).astype(np.float32)
        prob_of_zero = 0.5
        rx = np.where(np.greater(rand, prob_of_zero), rx, np.zeros_like(rx))
        ry = np.where(np.greater(rand, prob_of_zero), ry, np.zeros_like(ry))
        rz = np.where(np.greater(rand, prob_of_zero), rz, np.zeros_like(rz))
        tx = np.where(np.greater(rand, prob_of_zero), tx, np.zeros_like(tx))
        ty = np.where(np.greater(rand, prob_of_zero), ty, np.zeros_like(ty))
        tz = np.where(np.greater(rand, prob_of_zero), tz, np.zeros_like(tz))
        
    t = np.stack([tx, ty, tz], axis=1)
    t = torch.from_numpy(t)
    rx = torch.from_numpy(rx)
    ry = torch.from_numpy(ry)
    rz = torch.from_numpy(rz)
    r = eul2rotm(rx, ry, rz)
    rt = merge_rt(r, t)
    return rt

# def convert_boxlist_to_rtlist(boxlist):
#     B, N, D = list(boxlist.shape)
#     boxlist_ = boxlist.view(B*N, D)
#     rtlist_ = convert_box_to_ref_T_obj(boxlist_)
#     rtlist = rtlist_.view(B, N, 4, 4)
#     return rtlist
    
def convert_boxlist_to_lrtlist(boxlist):
    B, N, D = list(boxlist.shape)
    boxlist_ = boxlist.view(B*N, D)
    rtlist_ = convert_box_to_ref_T_obj(boxlist_)
    rtlist = rtlist_.view(B, N, 4, 4)
    lenlist = boxlist[:,:,3:6].reshape(B, N, 3)
    lrtlist = merge_lrtlist(lenlist, rtlist)
    return lrtlist
    
def convert_box_to_ref_T_obj(box3D):
    # turn the box into obj_T_ref (i.e., obj_T_cam)
    B = list(box3D.shape)[0]
    
    # box3D is B x 9
    x, y, z, lx, ly, lz, rx, ry, rz = torch.unbind(box3D, axis=1)
    rot0 = eye_3x3(B)
    tra = torch.stack([x, y, z], axis=1)
    center_T_ref = merge_rt(rot0, -tra)
    # center_T_ref is B x 4 x 4
    
    t0 = torch.zeros([B, 3])
    rot = eul2rotm(rx, -ry, -rz)
    obj_T_center = merge_rt(rot, t0)
    # this is B x 4 x 4

    # we want obj_T_ref
    # first we to translate to center,
    # and then rotate around the origin
    obj_T_ref = utils_basic.matmul2(obj_T_center, center_T_ref)

    # return the inverse of this, so that we can transform obj corners into cam coords
    ref_T_obj = obj_T_ref.inverse()
    return ref_T_obj

def get_xyzlist_from_lenlist(lenlist):
    B, N, D = list(lenlist.shape)
    assert(D==3)
    lx, ly, lz = torch.unbind(lenlist, axis=2)
    xs = torch.stack([-lx/2., -lx/2., -lx/2., -lx/2., lx/2., lx/2., lx/2., lx/2.], axis=2)
    ys = torch.stack([-ly/2., -ly/2., ly/2., ly/2., -ly/2., -ly/2., ly/2., ly/2.], axis=2)
    zs = torch.stack([-lz/2., lz/2., -lz/2., lz/2., -lz/2., lz/2., -lz/2., lz/2.], axis=2)
    # these are B x N x 8
    xyzlist = torch.stack([xs, ys, zs], axis=3)
    # this is B x N x 8 x 3
    return xyzlist

def transform_boxes_to_corners_single(boxes):
    N, D = list(boxes.shape)
    assert(D==9)
    
    xc,yc,zc,lx,ly,lz,rx,ry,rz = torch.unbind(boxes, axis=1)
    # these are each shaped N

    ref_T_obj = convert_box_to_ref_T_obj(boxes)

    xs = torch.stack([-lx/2., -lx/2., -lx/2., -lx/2., lx/2., lx/2., lx/2., lx/2.], axis=1)
    ys = torch.stack([-ly/2., -ly/2., ly/2., ly/2., -ly/2., -ly/2., ly/2., ly/2.], axis=1)
    zs = torch.stack([-lz/2., lz/2., -lz/2., lz/2., -lz/2., lz/2., -lz/2., lz/2.], axis=1)
    
    xyz_obj = torch.stack([xs, ys, zs], axis=2)
    # centered_box is N x 8 x 3

    xyz_ref = apply_4x4(ref_T_obj, xyz_obj)
    # xyz_ref is N x 8 x 3
    return xyz_ref

def eul2rotm_py(rx, ry, rz):
    # inputs are shaped B
    # this func is copied from matlab
    # R = [  cy*cz   sy*sx*cz-sz*cx    sy*cx*cz+sz*sx
    #        cy*sz   sy*sx*sz+cz*cx    sy*cx*sz-cz*sx
    #        -sy            cy*sx             cy*cx]
    rx = rx[:,np.newaxis]
    ry = ry[:,np.newaxis]
    rz = rz[:,np.newaxis]
    # these are B x 1
    sinz = np.sin(rz)
    siny = np.sin(ry)
    sinx = np.sin(rx)
    cosz = np.cos(rz)
    cosy = np.cos(ry)
    cosx = np.cos(rx)
    r11 = cosy*cosz
    r12 = sinx*siny*cosz - cosx*sinz
    r13 = cosx*siny*cosz + sinx*sinz
    r21 = cosy*sinz
    r22 = sinx*siny*sinz + cosx*cosz
    r23 = cosx*siny*sinz - sinx*cosz
    r31 = -siny
    r32 = sinx*cosy
    r33 = cosx*cosy
    r1 = np.stack([r11,r12,r13],axis=2)
    r2 = np.stack([r21,r22,r23],axis=2)
    r3 = np.stack([r31,r32,r33],axis=2)
    r = np.concatenate([r1,r2,r3],axis=1)
    return r

def apply_4x4_py(RT, XYZ):
    # RT is B x 4 x 4
    # XYZ is B x N x 3

    # put into homogeneous coords
    X, Y, Z = np.split(XYZ, 3, axis=2)
    ones = np.ones_like(X)
    XYZ1 = np.concatenate([X, Y, Z, ones], axis=2)
    # XYZ1 is B x N x 4

    XYZ1_t = np.transpose(XYZ1, (0,2,1))
    # this is B x 4 x N

    XYZ2_t = np.matmul(RT, XYZ1_t)
    # this is B x 4 x N
    
    XYZ2 = np.transpose(XYZ2_t, (0,2,1))
    # this is B x N x 4
    
    XYZ2 = XYZ2[:,:,:3]
    # this is B x N x 3
    
    return XYZ2


def merge_rt_py(r, t):
    # r is B x 3 x 3
    # t is B x 3

    if r is None and t is None:
        assert(False) # you have to provide either r or t
        
    if r is None:
        shape = t.shape
        B = int(shape[0])
        r = np.tile(np.eye(3)[np.newaxis,:,:], (B,1,1))
    elif t is None:
        shape = r.shape
        B = int(shape[0])
        
        t = np.zeros((B, 3))
    else:
        shape = r.shape
        B = int(shape[0])
        
    bottom_row = np.tile(np.reshape(np.array([0.,0.,0.,1.], dtype=np.float32),[1,1,4]),
                         [B,1,1])
    rt = np.concatenate([r,np.expand_dims(t,2)], axis=2)
    rt = np.concatenate([rt,bottom_row], axis=1)
    return rt


def transform_boxes3D_to_corners_py(boxes3D):
    N, D = list(boxes3D.shape)
    assert(D==9)
    
    xc,yc,zc,lx,ly,lz,rx,ry,rz = boxes3D[:,0], boxes3D[:,1], boxes3D[:,2], boxes3D[:,3], boxes3D[:,4], boxes3D[:,5], boxes3D[:,6], boxes3D[:,7], boxes3D[:,8]

    # these are each shaped N

    rotation_mat = eul2rotm_py(rx, ry, rz)
    translation = np.stack([xc, yc, zc], axis=1) 
    ref_T_obj = merge_rt_py(rotation_mat, translation)

    xs = np.stack([lx/2., lx/2., -lx/2., -lx/2., lx/2., lx/2., -lx/2., -lx/2.], axis=1)
    ys = np.stack([ly/2., ly/2., ly/2., ly/2., -ly/2., -ly/2., -ly/2., -ly/2.], axis=1)
    zs = np.stack([lz/2., -lz/2., -lz/2., lz/2., lz/2., -lz/2., -lz/2., lz/2.], axis=1)

    # xs = tf.stack([-lx/2., -lx/2., lx/2., lx/2., -lx/2., -lx/2., lx/2., lx/2.], axis=1)
    # ys = tf.stack([ly/2., -ly/2., ly/2., -ly/2., ly/2., -ly/2., ly/2., -ly/2.], axis=1)
    # zs = tf.stack([-lz/2., -lz/2., -lz/2., -lz/2., lz/2., lz/2., lz/2., lz/2.], axis=1)

    xyz_obj = np.stack([xs, ys, zs], axis=2)
    # centered_box is N x 8 x 3

    xyz_ref = apply_4x4_py(ref_T_obj, xyz_obj)
    # xyz_ref is N x 8 x 3
    return xyz_ref


def transform_boxes_to_corners(boxes):
    # returns corners, shaped B x N x 8 x 3
    B, N, D = list(boxes.shape)
    assert(D==9)
    
    __p = lambda x: utils_basic.pack_seqdim(x, B)
    __u = lambda x: utils_basic.unpack_seqdim(x, B)

    boxes_ = __p(boxes)
    corners_ = transform_boxes_to_corners_single(boxes_)
    corners = __u(corners_)
    return corners


def transform_corners_to_boxes(corners):
    # corners is B x N x 8 x 3
    B, N, C, D = corners.shape
    assert(C==8)
    assert(D==3)
    # do them all at once
    __p = lambda x: utils_basic.pack_seqdim(x, B)
    __u = lambda x: utils_basic.unpack_seqdim(x, B)
    corners_ = __p(corners)
    boxes_ = transform_corners_to_boxes_single(corners_)
    boxes_ = boxes_.cuda()
    boxes = __u(boxes_)
    return boxes

def transform_corners_to_boxes_single(corners):
    # corners is B x 8 x 3
    corners = corners.detach().cpu().numpy()

    # assert(False) # this function has a flaw; use rigid_transform_boxes instead, or fix it.
    # # i believe you can fix it using what i noticed in rigid_transform_boxes:
    # # if we are looking at the box backwards, the rx/rz dirs flip

    # we want to transform each one to a box
    # note that the rotation may flip 180deg, since corners do not have this info
    
    boxes = []
    for ind, corner_set in enumerate(corners):
        xs = corner_set[:,0]
        ys = corner_set[:,1]
        zs = corner_set[:,2]
        # these are 8 each

        xc = np.mean(xs)
        yc = np.mean(ys)
        zc = np.mean(zs)

        # we constructed the corners like this:
        # xs = tf.stack([-lx/2., -lx/2., -lx/2., -lx/2., lx/2., lx/2., lx/2., lx/2.], axis=1)
        # ys = tf.stack([-ly/2., -ly/2., ly/2., ly/2., -ly/2., -ly/2., ly/2., ly/2.], axis=1)
        # zs = tf.stack([-lz/2., lz/2., -lz/2., lz/2., -lz/2., lz/2., -lz/2., lz/2.], axis=1)
        # # so we can recover lengths like this:
        # lx = np.linalg.norm(xs[-1] - xs[0])
        # ly = np.linalg.norm(ys[-1] - ys[0])
        # lz = np.linalg.norm(zs[-1] - zs[0])
        # but that's a noisy estimate apparently. let's try all pairs

        # rotations are a bit more interesting...

        # defining the corners as: clockwise backcar face, clockwise frontcar face:
        #   E -------- F
        #  /|         /|
        # A -------- B .
        # | |        | |
        # . H -------- G
        # |/         |/
        # D -------- C

        # the ordered eight indices are:
        # A E D H B F C G

        # unstack on first dim
        A, E, D, H, B, F, C, G = corner_set

        back = [A, B, C, D] # back of car is closer to us
        front = [E, F, G, H]
        top = [A, E, B, F]
        bottom = [D, C, H, G]

        front = np.stack(front, axis=0)
        back = np.stack(back, axis=0)
        top = np.stack(top, axis=0)
        bottom = np.stack(bottom, axis=0)
        # these are 4 x 3

        back_z = np.mean(back[:,2])
        front_z = np.mean(front[:,2])
        # usually the front has bigger coords than back
        backwards = not (front_z > back_z)

        front_y = np.mean(front[:,1])
        back_y = np.mean(back[:,1])
        # someetimes the front dips down
        dips_down = front_y > back_y

        # the bottom should have bigger y coords than the bottom (since y increases down)
        top_y = np.mean(top[:,2])
        bottom_y = np.mean(bottom[:,2])
        upside_down = not (top_y < bottom_y)

        # rx: i need anything but x-aligned bars
        # there are 8 of these
        # atan2 wants the y part then the x part; here this means y then z

        x_bars = [[A, B], [D, C], [E, F], [H, G]]
        y_bars = [[A, D], [B, C], [E, H], [F, G]]
        z_bars = [[A, E], [B, F], [D, H], [C, G]]

        lx = 0.0
        for x_bar in x_bars:
            x0, x1 = x_bar
            lx += np.linalg.norm(x1-x0)
        lx /= 4.0

        ly = 0.0
        for y_bar in y_bars:
            y0, y1 = y_bar
            ly += np.linalg.norm(y1-y0)
        ly /= 4.0

        lz = 0.0
        for z_bar in z_bars:
            z0, z1 = z_bar
            lz += np.linalg.norm(z1-z0)
        lz /= 4.0
        rx = 0.0
        for bar in z_bars:
            pt1, pt2 = bar
            intermed = np.arctan2((pt1[1] - pt2[1]), (pt1[2] - pt2[2]))
            rx += intermed

        rx /= 4.0

        ry = 0.0
        for bar in z_bars:
            pt1, pt2 = bar
            intermed = np.arctan2((pt1[2] - pt2[2]), (pt1[0] - pt2[0]))
            ry += intermed

        ry /= 4.0

        rz = 0.0
        for bar in x_bars:
            pt1, pt2 = bar
            intermed = np.arctan2((pt1[1] - pt2[1]), (pt1[0] - pt2[0]))
            rz += intermed

        rz /= 4.0

        ry += np.pi/2.0

        if backwards:
            ry = -ry
        if not backwards:
            ry = ry - np.pi

        box = np.array([xc, yc, zc, lx, ly, lz, rx, ry, rz])
        boxes.append(box)
    boxes = np.stack(boxes, axis=0).astype(np.float32)
    return torch.from_numpy(boxes)
    


def apply_pix_T_cam(pix_T_cam, xyz):

    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    
    # xyz is shaped B x H*W x 3
    # returns xy, shaped B x H*W x 2
    
    B, N, C = list(xyz.shape)
    assert(C==3)
    
    x, y, z = torch.unbind(xyz, axis=-1)

    fx = torch.reshape(fx, [B, 1])
    fy = torch.reshape(fy, [B, 1])
    x0 = torch.reshape(x0, [B, 1])
    y0 = torch.reshape(y0, [B, 1])

    EPS=1e-6
    x = (x*fx)/(z+EPS)+x0
    y = (y*fy)/(z+EPS)+y0
    xy = torch.stack([x, y], axis=-1)
    return xy

# def apply_4x4_to_boxes(Y_T_X, boxes_X):
#     B, N, C = boxes_X.get_shape().as_list()
#     assert(C==9)
#     corners_X = transform_boxes_to_corners(boxes_X) # corners is B x N x 8 x 3
#     corners_X_ = tf.reshape(corners_X, [B, N*8, 3])
#     corners_Y_ = apply_4x4(Y_T_X, corners_X_)
#     corners_Y = tf.reshape(corners_Y_, [B, N, 8, 3])
#     boxes_Y = corners_to_boxes(corners_Y)
#     return boxes_Y

def apply_4x4_to_corners(Y_T_X, corners_X):
    B, N, C, D = list(corners_X.shape)
    assert(C==8)
    assert(D==3)
    corners_X_ = torch.reshape(corners_X, [B, N*8, 3])
    corners_Y_ = apply_4x4(Y_T_X, corners_X_)
    corners_Y = torch.reshape(corners_Y_, [B, N, 8, 3])
    return corners_Y

def split_lrt(lrt):
    # splits a B x 19 tensor
    # into B x 3 (lens)
    # and B x 4 x 4 (rts)
    B, D = list(lrt.shape)
    assert(D==19)
    lrt = lrt.unsqueeze(1)
    l, rt = split_lrtlist(lrt)
    l = l.squeeze(1)
    rt = rt.squeeze(1)
    return l, rt

def split_lrtlist(lrtlist):
    # splits a B x N x 19 tensor
    # into B x N x 3 (lens)
    # and B x N x 4 x 4 (rts)
    B, N, D = list(lrtlist.shape)
    assert(D==19)
    lenlist = lrtlist[:,:,:3].reshape(B, N, 3)
    ref_T_objs_list = lrtlist[:,:,3:].reshape(B, N, 4, 4)
    return lenlist, ref_T_objs_list

def merge_lrtlist(lenlist, rtlist):
    # lenlist is B x N x 3
    # rtlist is B x N x 4 x 4
    # merges these into a B x N x 19 tensor
    B, N, D = list(lenlist.shape)
    assert(D==3)
    B2, N2, E, F = list(rtlist.shape)
    assert(B==B2)
    assert(N==N2)
    assert(E==4 and F==4)
    rtlist = rtlist.reshape(B, N, 16)
    lrtlist = torch.cat([lenlist, rtlist], axis=2)
    return lrtlist

def apply_4x4_to_lrtlist(Y_T_X, lrtlist_X):
    B, N, D = list(lrtlist_X.shape)
    assert(D==19)
    B2, E, F = list(Y_T_X.shape)
    assert(B2==B)
    assert(E==4 and F==4)
    
    lenlist, rtlist_X = split_lrtlist(lrtlist_X)
    # rtlist_X is B x N x 4 x 4

    Y_T_Xs = Y_T_X.unsqueeze(1).repeat(1, N, 1, 1)
    Y_T_Xs_ = Y_T_Xs.view(B*N, 4, 4)
    rtlist_X_ = rtlist_X.view(B*N, 4, 4)
    rtlist_Y_ = utils_basic.matmul2(Y_T_Xs_, rtlist_X_)
    rtlist_Y = rtlist_Y_.view(B, N, 4, 4)
    lrtlist_Y = merge_lrtlist(lenlist, rtlist_Y)
    return lrtlist_Y

# import time
# if __name__ == "__main__":
#     input = torch.rand(10, 4, 4).cuda()
#     cur_time = time.time()
#     out_1 = safe_inverse(input)
#     print('time for non-parallel:{}'.format(time.time() - cur_time))

#     print(out_1[0])

#     cur_time = time.time()
#     out_2 = safe_inverse_parallel(input)
#     print('time for parallel:{}'.format(time.time() - cur_time))

#     print(out_2[0])

def create_depth_image_single(xy, z, H, W):
    # turn the xy coordinates into image inds
    xy = torch.round(xy).long()
    depth = torch.zeros(H*W, dtype=torch.float32, device=torch.device('cuda'))
    
    # lidar reports a sphere of measurements
    # only use the inds that are within the image bounds
    # also, only use forward-pointing depths (z > 0)
    valid = (xy[:,0] <= W-1) & (xy[:,1] <= H-1) & (xy[:,0] >= 0) & (xy[:,1] >= 0) & (z[:] > 0)

    # gather these up
    xy = xy[valid]
    z = z[valid]

    inds = utils_basic.sub2ind(H, W, xy[:,1], xy[:,0]).long()
    depth[inds] = z
    valid = (depth > 0.0).float()
    depth[torch.where(depth == 0.0)] = 100.0
    depth = torch.reshape(depth, [1, H, W])
    valid = torch.reshape(valid, [1, H, W])
    return depth, valid

def create_depth_image(pix_T_cam, xyz_cam, H, W):
    B, N, D = list(xyz_cam.shape)
    assert(D==3)
    xy = apply_pix_T_cam(pix_T_cam, xyz_cam)
    z = xyz_cam[:,:,2]

    depth = torch.zeros(B, 1, H, W, dtype=torch.float32, device=torch.device('cuda'))
    valid = torch.zeros(B, 1, H, W, dtype=torch.float32, device=torch.device('cuda'))
    for b in range(B):
        depth[b], valid[b] = create_depth_image_single(xy[b], z[b], H, W)
    return depth, valid
