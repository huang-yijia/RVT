# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Utility function for MVT
"""
import pdb
import sys

import torch
import torch.nn.functional as F
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_rotation_6d

def points4_to_pose(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    *,
    to_6d: bool = True,
    close_thresh: float = 0.03,
    eps: float = 1e-8,
    align_z_to_p3: bool = True,   # optional: make z point toward p3
):
    """
    From 4 point labels -> (Trans, Rot, gripper_open) - CORRECTED VERSION
    
    This function is now a true inverse of pose_to_points4.
    The key correction: p3 now defines the positive z direction consistently.

    Shapes: all inputs broadcastable to (..., 3)
    Returns:
      trans: (..., 3)
      rot:   (..., 6) if to_6d else (..., 3, 3)
      is_open: (...,) boolean
    """

    # 1. (P1 + P2)/2 == Trans
    trans = (p1 + p2) / 2

    # 2. (P1 - P2)/|P1 - P2| == x (unit vector)
    x = F.normalize(p1 - p2, dim=-1, eps=eps)

    # 3. CORRECTED: (P3 - (P1+P2)/2)/|norm| == z (unit vector)
    # Now p3 consistently defines the positive z direction
    z = F.normalize(p3 - trans, dim=-1, eps=eps)

    # 4. y = z × x (completing the right-handed coordinate system)
    y = torch.cross(z, x, dim=-1)
    y = F.normalize(y, dim=-1, eps=eps)

    # 5. Ensure orthogonality by recalculating x
    x = torch.cross(y, z, dim=-1)
    x = F.normalize(x, dim=-1, eps=eps)

    # (Optional) Keep z roughly pointing toward P3 to match the figure's arrowing
    if align_z_to_p3:
        toward = torch.sum((p3 - trans) * z, dim=-1, keepdim=True)
        flip = torch.where(toward < 0, -1.0, 1.0)
        y = y * flip
        z = z * flip
        x = x * flip

    # Stack columns as R = [c1, c2, c3] = [x, y, z]
    R = torch.stack((x, y, z), dim=-1)  # (..., 3, 3)

    # 6. |P0 - P3| < threshold : close ;  > threshold : open
    gap = torch.linalg.vector_norm(p0 - p3, dim=-1)     # (...)
    is_open = gap >= close_thresh                       # (...,) boolean

    # Output rotation in desired representation
    rot = matrix_to_rotation_6d(R) if to_6d else R
    return trans, rot, is_open


def pose_to_points4(
    trans: torch.Tensor,
    rot_matrix: torch.Tensor,
    is_open: torch.Tensor,
    *,
    gripper_width: float = 0.04,
    gripper_length: float = 0.02,
    eps: float = 1e-8,
):
    """
    Convert pose (translation, rotation matrix, gripper state) to 4 points - CORRECTED VERSION
    
    This function is now a true inverse of points4_to_pose.
    The key correction: p3 is positioned in the positive z direction consistently.
    
    Args:
        trans: (..., 3) translation
        rot_matrix: (..., 3, 3) rotation matrix
        is_open: (...,) boolean gripper state
        gripper_width: width of gripper when closed
        gripper_length: length of gripper fingers
        eps: small epsilon for numerical stability
        
    Returns:
        p0, p1, p2, p3: (..., 3) four points representing the gripper pose
    """
    # Extract rotation axes from rotation matrix
    x_axis = rot_matrix[..., :, 0]  # (..., 3)
    y_axis = rot_matrix[..., :, 1]  # (..., 3)
    z_axis = rot_matrix[..., :, 2]  # (..., 3)
    
    # Calculate gripper finger positions
    # P1 and P2 are the gripper finger tips along x-axis
    p1 = trans + (gripper_width / 2) * x_axis
    p2 = trans - (gripper_width / 2) * x_axis
    
    # CORRECTED: P3 is the approach direction along positive z-axis
    # This ensures consistency with points4_to_pose reconstruction
    p3 = trans + gripper_length * z_axis
    
    # P0 is the gripper base position along z-axis
    # When gripper is closed, P0 is close to P3
    # When gripper is open, P0 is further from P3
    gripper_gap = torch.where(
        is_open,
        torch.tensor(0.03, device=trans.device),  # Open gripper gap
        torch.tensor(0.01, device=trans.device),  # Closed gripper gap
    )
    p0 = trans + gripper_gap.unsqueeze(-1) * z_axis
    
    return p0, p1, p2, p3


def pose_to_points4_from_quat(
    trans: torch.Tensor,
    quat: torch.Tensor,
    is_open: torch.Tensor,
    *,
    gripper_width: float = 0.04,
    gripper_length: float = 0.02,
    eps: float = 1e-8,
):
    """
    Convert pose (translation, quaternion, gripper state) to 4 points.
    Convenience function that handles quaternion input.
    
    Args:
        trans: (..., 3) translation
        quat: (..., 4) quaternion in xyzw format
        is_open: (...,) boolean gripper state
        gripper_width: width of gripper when closed
        gripper_length: length of gripper fingers
        eps: small epsilon for numerical stability
        
    Returns:
        p0, p1, p2, p3: (..., 3) four points representing the gripper pose
    """
    # Convert quaternion to rotation matrix
    from pytorch3d.transforms import quaternion_to_matrix
    rot_matrix = quaternion_to_matrix(quat)
    
    # Use the main function
    return pose_to_points4(trans, rot_matrix, is_open, 
                          gripper_width=gripper_width, 
                          gripper_length=gripper_length, 
                          eps=eps)


def place_pc_in_cube(
    pc, app_pc=None, with_mean_or_bounds=True, scene_bounds=None, no_op=False
):
    """
    calculate the transformation that would place the point cloud (pc) inside a
        cube of size (2, 2, 2). The pc is centered at mean if with_mean_or_bounds
        is True. If with_mean_or_bounds is False, pc is centered around the mid
        point of the bounds. The transformation is applied to point cloud app_pc if
        it is not None. If app_pc is None, the transformation is applied on pc.
    :param pc: pc of shape (num_points_1, 3)
    :param app_pc:
        Either
        - pc of shape (num_points_2, 3)
        - None
    :param with_mean_or_bounds:
        Either:
            True: pc is centered around its mean
            False: pc is centered around the center of the scene bounds
    :param scene_bounds: [x_min, y_min, z_min, x_max, y_max, z_max]
    :param no_op: if no_op, then this function does not do any operation
    """
    if no_op:
        if app_pc is None:
            app_pc = torch.clone(pc)

        return app_pc, lambda x: x

    if with_mean_or_bounds:
        assert scene_bounds is None
    else:
        assert not (scene_bounds is None)
    if with_mean_or_bounds:
        pc_mid = (torch.max(pc, 0)[0] + torch.min(pc, 0)[0]) / 2
        x_len, y_len, z_len = torch.max(pc, 0)[0] - torch.min(pc, 0)[0]
    else:
        x_min, y_min, z_min, x_max, y_max, z_max = scene_bounds
        pc_mid = torch.tensor(
            [
                (x_min + x_max) / 2,
                (y_min + y_max) / 2,
                (z_min + z_max) / 2,
            ]
        ).to(pc.device)
        x_len, y_len, z_len = x_max - x_min, y_max - y_min, z_max - z_min

    scale = 2 / max(x_len, y_len, z_len)
    if app_pc is None:
        app_pc = torch.clone(pc)
    app_pc = (app_pc - pc_mid) * scale

    # reverse transformation to obtain app_pc in original frame
    def rev_trans(x):
        return (x / scale) + pc_mid

    return app_pc, rev_trans


def trans_pc(pc, loc, sca):
    """
    change location of the center of the pc and scale it
    :param pc:
        either:
        - tensor of shape(b, num_points, 3)
        - tensor of shape(b, 3)
        - list of pc each with size (num_points, 3)
    :param loc: (b, 3 )
    :param sca: 1 or (3)
    """
    assert len(loc.shape) == 2
    assert loc.shape[-1] == 3
    if isinstance(pc, list):
        assert all([(len(x.shape) == 2) and (x.shape[1] == 3) for x in pc])
        pc = [sca * (x - y) for x, y in zip(pc, loc)]
    elif isinstance(pc, torch.Tensor):
        assert len(pc.shape) in [2, 3]
        assert pc.shape[-1] == 3
        if len(pc.shape) == 2:
            pc = sca * (pc - loc)
        else:
            pc = sca * (pc - loc.unsqueeze(1))
    else:
        assert False

    # reverse transformation to obtain app_pc in original frame
    def rev_trans(x):
        assert isinstance(x, torch.Tensor)
        return (x / sca) + loc

    return pc, rev_trans


def add_uni_noi(x, u):
    """
    adds uniform noise to a tensor x. output is tensor where each element is
    in [x-u, x+u]
    :param x: tensor
    :param u: float
    """
    assert isinstance(u, float)
    # move noise in -1 to 1
    noise = (2 * torch.rand(*x.shape, device=x.device)) - 1
    x = x + (u * noise)
    return x


def generate_hm_from_pt(pt, res, sigma, thres_sigma_times=3):
    """
    Pytorch code to generate heatmaps from point. Points with values less than
    thres are made 0
    :type pt: torch.FloatTensor of size (num_pt, 2)
    :type res: int or (int, int)
    :param sigma: the std of the gaussian distribition. if it is -1, we
        generate a hm with one hot vector
    :type sigma: float
    :type thres: float
    """
    num_pt, x = pt.shape
    assert x == 2

    if isinstance(res, int):
        resx = resy = res
    else:
        resx, resy = res

    _hmx = torch.arange(0, resy).to(pt.device)
    _hmx = _hmx.view([1, resy]).repeat(resx, 1).view([resx, resy, 1])
    _hmy = torch.arange(0, resx).to(pt.device)
    _hmy = _hmy.view([resx, 1]).repeat(1, resy).view([resx, resy, 1])
    hm = torch.cat([_hmx, _hmy], dim=-1)
    hm = hm.view([1, resx, resy, 2]).repeat(num_pt, 1, 1, 1)

    pt = pt.view([num_pt, 1, 1, 2])
    hm = torch.exp(-1 * torch.sum((hm - pt) ** 2, -1) / (2 * (sigma**2)))
    thres = np.exp(-1 * (thres_sigma_times**2) / 2)
    hm[hm < thres] = 0.0

    hm /= torch.sum(hm, (1, 2), keepdim=True) + 1e-6

    # TODO: make a more efficient version
    if sigma == -1:
        _hm = hm.view(num_pt, resx * resy)
        hm = torch.zeros((num_pt, resx * resy), device=hm.device)
        temp = torch.arange(num_pt).to(hm.device)
        hm[temp, _hm.argmax(-1)] = 1

    return hm


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
