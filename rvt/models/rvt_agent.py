# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import pprint

import clip
import torch
import torchvision
import numpy as np
import torch.nn as nn
import bitsandbytes as bnb

from scipy.spatial.transform import Rotation
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch3d import transforms as torch3d_tf

import rvt.utils.peract_utils as peract_utils
import rvt.mvt.utils as mvt_utils
import rvt.utils.rvt_utils as rvt_utils
import peract_colab.arm.utils as arm_utils

from rvt.mvt.augmentation import apply_se3_aug_con, aug_utils
from peract_colab.arm.optim.lamb import Lamb
from yarr.agents.agent import ActResult
from rvt.utils.dataset import _clip_encode_text
from rvt.utils.lr_sched_utils import GradualWarmupScheduler


def eval_con(gt, pred):
    assert gt.shape == pred.shape, print(f"{gt.shape} {pred.shape}")
    assert len(gt.shape) == 2
    dist = torch.linalg.vector_norm(gt - pred, dim=1)
    return {"avg err": dist.mean()}


def eval_con_cls(gt, pred, num_bin=72, res=5, symmetry=1):
    """
    Evaluate continuous classification where floating point values are put into
    discrete bins
    :param gt: (bs,)
    :param pred: (bs,)
    :param num_bin: int for the number of rotation bins
    :param res: float to specify the resolution of each rotation bin
    :param symmetry: degrees of symmetry; 2 is 180 degree symmetry, 4 is 90
        degree symmetry
    """
    assert gt.shape == pred.shape
    assert len(gt.shape) in [0, 1], gt
    assert num_bin % symmetry == 0, (num_bin, symmetry)
    gt = torch.tensor(gt)
    pred = torch.tensor(pred)
    num_bin //= symmetry
    pred %= num_bin
    gt %= num_bin
    dist = torch.abs(pred - gt)
    dist = torch.min(dist, num_bin - dist)
    dist_con = dist.float() * res
    return {"avg err": dist_con.mean()}


def eval_cls(gt, pred):
    """
    Evaluate classification performance
    :param gt_coll: (bs,)
    :param pred: (bs,)
    """
    assert gt.shape == pred.shape
    assert len(gt.shape) == 1
    return {"per err": (gt != pred).float().mean()}


def eval_all(
    wpt,
    pred_wpt,
    action_rot,
    pred_rot_quat,
    action_grip_one_hot,
    grip_q,
    action_collision_one_hot,
    collision_q,
):
    bs = len(wpt)
    assert wpt.shape == (bs, 3), wpt
    # pred_wpt now has shape (bs, 4, 3) - use only the first waypoint for evaluation
    if len(pred_wpt.shape) == 3:
        pred_wpt = pred_wpt[:, 0, :]  # Take only the first waypoint (bs, 3)
    assert pred_wpt.shape == (bs, 3), pred_wpt
    assert action_rot.shape == (bs, 4), action_rot
    assert pred_rot_quat.shape == (bs, 4), pred_rot_quat
    assert action_grip_one_hot.shape == (bs, 2), action_grip_one_hot
    assert grip_q.shape == (bs, 2), grip_q
    assert action_collision_one_hot.shape == (bs, 2), action_collision_one_hot
    assert collision_q.shape == (bs, 2), collision_q

    eval_trans = []
    eval_rot_x = []
    eval_rot_y = []
    eval_rot_z = []
    eval_grip = []
    eval_coll = []

    for i in range(bs):
        eval_trans.append(
            eval_con(wpt[i : i + 1], pred_wpt[i : i + 1])["avg err"]
            .cpu()
            .numpy()
            .item()
        )

        euler_gt = Rotation.from_quat(action_rot[i]).as_euler("xyz", degrees=True)
        euler_pred = Rotation.from_quat(pred_rot_quat[i]).as_euler("xyz", degrees=True)

        eval_rot_x.append(
            eval_con_cls(euler_gt[0], euler_pred[0], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )
        eval_rot_y.append(
            eval_con_cls(euler_gt[1], euler_pred[1], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )
        eval_rot_z.append(
            eval_con_cls(euler_gt[2], euler_pred[2], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )

        eval_grip.append(
            eval_cls(
                action_grip_one_hot[i : i + 1].argmax(-1),
                grip_q[i : i + 1].argmax(-1),
            )["per err"]
            .cpu()
            .numpy()
            .item()
        )

        eval_coll.append(
            eval_cls(
                action_collision_one_hot[i : i + 1].argmax(-1),
                collision_q[i : i + 1].argmax(-1),
            )["per err"]
            .cpu()
            .numpy()
        )

    return eval_trans, eval_rot_x, eval_rot_y, eval_rot_z, eval_grip, eval_coll


def manage_eval_log(
    self,
    tasks,
    wpt,
    pred_wpt,
    action_rot,
    pred_rot_quat,
    action_grip_one_hot,
    grip_q,
    action_collision_one_hot,
    collision_q,
    reset_log=False,
):
    bs = len(wpt)
    assert wpt.shape == (bs, 3), wpt
    # pred_wpt now has shape (bs, 4, 3) - use only the first waypoint for evaluation
    if len(pred_wpt.shape) == 3:
        pred_wpt = pred_wpt[:, 0, :]  # Take only the first waypoint (bs, 3)
    assert pred_wpt.shape == (bs, 3), pred_wpt
    assert action_rot.shape == (bs, 4), action_rot
    assert pred_rot_quat.shape == (bs, 4), pred_rot_quat
    assert action_grip_one_hot.shape == (bs, 2), action_grip_one_hot
    assert grip_q.shape == (bs, 2), grip_q
    assert action_collision_one_hot.shape == (bs, 2), action_collision_one_hot
    assert collision_q.shape == (bs, 2), collision_q

    if not hasattr(self, "eval_trans") or reset_log:
        self.eval_trans = {}
        self.eval_rot_x = {}
        self.eval_rot_y = {}
        self.eval_rot_z = {}
        self.eval_grip = {}
        self.eval_coll = {}

    (eval_trans, eval_rot_x, eval_rot_y, eval_rot_z, eval_grip, eval_coll,) = eval_all(
        wpt=wpt,
        pred_wpt=pred_wpt,
        action_rot=action_rot,
        pred_rot_quat=pred_rot_quat,
        action_grip_one_hot=action_grip_one_hot,
        grip_q=grip_q,
        action_collision_one_hot=action_collision_one_hot,
        collision_q=collision_q,
    )

    for idx, task in enumerate(tasks):
        if not (task in self.eval_trans):
            self.eval_trans[task] = []
            self.eval_rot_x[task] = []
            self.eval_rot_y[task] = []
            self.eval_rot_z[task] = []
            self.eval_grip[task] = []
            self.eval_coll[task] = []
        self.eval_trans[task].append(eval_trans[idx])
        self.eval_rot_x[task].append(eval_rot_x[idx])
        self.eval_rot_y[task].append(eval_rot_y[idx])
        self.eval_rot_z[task].append(eval_rot_z[idx])
        self.eval_grip[task].append(eval_grip[idx])
        self.eval_coll[task].append(eval_coll[idx])

    return {
        "eval_trans": eval_trans,
        "eval_rot_x": eval_rot_x,
        "eval_rot_y": eval_rot_y,
        "eval_rot_z": eval_rot_z,
    }


def print_eval_log(self):
    logs = {
        "trans": self.eval_trans,
        "rot_x": self.eval_rot_x,
        "rot_y": self.eval_rot_y,
        "rot_z": self.eval_rot_z,
        "grip": self.eval_grip,
        "coll": self.eval_coll,
    }

    out = {}
    for name, log in logs.items():
        for task, task_log in log.items():
            task_log_np = np.array(task_log)
            mean, std, median = (
                np.mean(task_log_np),
                np.std(task_log_np),
                np.median(task_log_np),
            )
            out[f"{task}/{name}_mean"] = mean
            out[f"{task}/{name}_std"] = std
            out[f"{task}/{name}_median"] = median

    pprint.pprint(out)

    return out


def manage_loss_log(
    agent,
    loss_log,
    reset_log,
):
    if not hasattr(agent, "loss_log") or reset_log:
        agent.loss_log = {}

    for key, val in loss_log.items():
        if key in agent.loss_log:
            agent.loss_log[key].append(val)
        else:
            agent.loss_log[key] = [val]


def print_loss_log(agent):
    out = {}
    for key, val in agent.loss_log.items():
        out[key] = np.mean(np.array(val))
    pprint.pprint(out)
    return out


class RVTAgent:
    def __init__(
        self,
        network: nn.Module,
        num_rotation_classes: int, #72
        stage_two: bool,
        add_lang: bool,
        amp: bool,
        bnb: bool,
        move_pc_in_bound: bool,
        lr: float = 0.0001,
        lr_cos_dec: bool = False,
        cos_dec_max_step: int = 60000,
        warmup_steps: int = 0,
        image_resolution: list = None,
        lambda_weight_l2: float = 0.0,
        transform_augmentation: bool = True,
        transform_augmentation_xyz: list = [0.1, 0.1, 0.1],
        transform_augmentation_rpy: list = [0.0, 0.0, 20.0],
        place_with_mean: bool = True,
        transform_augmentation_rot_resolution: int = 5,
        optimizer_type: str = "lamb",
        gt_hm_sigma: float = 1.5,
        img_aug: bool = False,
        add_rgc_loss: bool = False,
        scene_bounds: list = peract_utils.SCENE_BOUNDS,
        cameras: list = peract_utils.CAMERAS,
        rot_ver: int = 0,
        rot_x_y_aug: int = 2,
        log_dir="",
        use_4point_pose: bool = False,
    ):
        """
        :param gt_hm_sigma: the std of the groundtruth hm, currently for for
            2d, if -1 then only single point is considered
        :type gt_hm_sigma: float
        :param rot_ver: version of the rotation prediction network
            Either:
                0: same as peract, independent discrete xyz predictions
                1: xyz prediction dependent on one another
        :param rot_x_y_aug: only applicable when rot_ver is 1, it specifies how
            much error we should add to groundtruth rotation while training
        :param log_dir: a folder location for saving some intermediate data
        """

        self._network = network
        self._num_rotation_classes = num_rotation_classes # 72
        self._rotation_resolution = 360 / self._num_rotation_classes # 5
        self._lr = lr
        self._image_resolution = image_resolution
        self._lambda_weight_l2 = lambda_weight_l2
        self._transform_augmentation = transform_augmentation
        self._place_with_mean = place_with_mean
        self._transform_augmentation_xyz = torch.from_numpy(
            np.array(transform_augmentation_xyz)
        )
        self._transform_augmentation_rpy = transform_augmentation_rpy
        self._transform_augmentation_rot_resolution = (
            transform_augmentation_rot_resolution
        )
        self._optimizer_type = optimizer_type
        self.gt_hm_sigma = gt_hm_sigma
        self.img_aug = img_aug
        self.add_rgc_loss = add_rgc_loss
        self.amp = amp
        self.bnb = bnb
        self.stage_two = stage_two
        self.add_lang = add_lang
        self.log_dir = log_dir
        self.warmup_steps = warmup_steps
        self.lr_cos_dec = lr_cos_dec
        self.cos_dec_max_step = cos_dec_max_step
        self.scene_bounds = scene_bounds
        self.cameras = cameras
        self.move_pc_in_bound = move_pc_in_bound
        self.rot_ver = rot_ver
        self.rot_x_y_aug = rot_x_y_aug
        self.use_4point_pose = use_4point_pose

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        if isinstance(self._network, DistributedDataParallel):
            self._net_mod = self._network.module
        else:
            self._net_mod = self._network

        self.num_all_rot = self._num_rotation_classes * 3 # 72 * 3 = 216

        self.scaler = GradScaler(enabled=self.amp)

    def build(self, training: bool, device: torch.device = None):
        self._training = training
        self._device = device

        if self._optimizer_type == "lamb":
            if self.bnb:
                print("Using 8-Bit Optimizer")
                self._optimizer = bnb.optim.LAMB(
                    self._network.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                    betas=(0.9, 0.999),
                )
            else:
                # From: https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py
                self._optimizer = Lamb(
                    self._network.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                    betas=(0.9, 0.999),
                    adam=False,
                )
        elif self._optimizer_type == "adam":
            self._optimizer = torch.optim.Adam(
                self._network.parameters(),
                lr=self._lr,
                weight_decay=self._lambda_weight_l2,
            )
        else:
            raise Exception("Unknown optimizer")

        if self.lr_cos_dec:
            after_scheduler = CosineAnnealingLR(
                self._optimizer,
                T_max=self.cos_dec_max_step,
                eta_min=self._lr / 100,  # mininum lr
            )
        else:
            after_scheduler = None
        self._lr_sched = GradualWarmupScheduler(
            self._optimizer,
            multiplier=1,
            total_epoch=self.warmup_steps,
            after_scheduler=after_scheduler,
        )

    def load_clip(self):
        self.clip_model, self.clip_preprocess = clip.load("RN50", device=self._device)
        self.clip_model.eval()

    def unload_clip(self):
        del self.clip_model
        del self.clip_preprocess
        with torch.cuda.device(self._device):
            torch.cuda.empty_cache()

    # copied from per-act and removed the translation part
    def _get_one_hot_expert_actions(
        self,
        batch_size,
        action_rot,
        action_grip,
        action_ignore_collisions,
        device,
    ):
        """_get_one_hot_expert_actions.

        :param batch_size: int
        :param action_rot: np.array of shape (bs, 4), quternion xyzw format
        :param action_grip: torch.tensor of shape (bs)
        :param action_ignore_collisions: torch.tensor of shape (bs)
        :param device:
        """
        bs = batch_size
        assert action_rot.shape == (bs, 4)
        assert action_grip.shape == (bs,), (action_grip, bs)

        action_rot_x_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_rot_y_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_rot_z_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_grip_one_hot = torch.zeros((bs, 2), dtype=int, device=device)
        action_collision_one_hot = torch.zeros((bs, 2), dtype=int, device=device)

        # fill one-hots
        for b in range(bs):
            gt_rot = action_rot[b]
            gt_rot = aug_utils.quaternion_to_discrete_euler(
                gt_rot, self._rotation_resolution
            )
            action_rot_x_one_hot[b, gt_rot[0]] = 1
            action_rot_y_one_hot[b, gt_rot[1]] = 1
            action_rot_z_one_hot[b, gt_rot[2]] = 1

            # grip
            gt_grip = action_grip[b]
            action_grip_one_hot[b, gt_grip] = 1

            # ignore collision
            gt_ignore_collisions = action_ignore_collisions[b, :]
            action_collision_one_hot[b, gt_ignore_collisions[0]] = 1

        return (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,
            action_collision_one_hot,
        )

    def get_q(self, out, dims, only_pred=False, get_q_trans=True):
        """
        :param out: output of mvt (b, 5, 220, 220)
        :param dims: tensor dimensions (bs, nc, h, w)
        :param only_pred: some speedupds if the q values are meant only for
            prediction
        :return: tuple of trans_q, rot_q, grip_q and coll_q that is used for
            training and preduction
        """
        bs, nc, h, w = dims
        assert isinstance(only_pred, bool)

        if get_q_trans:
            pts = None
            # out["trans"] has shape (bs, nc, trans_dim, h, w)
            # For original: trans_dim = 1, for 4-point: trans_dim = 4
            trans_dim = out["trans"].shape[2]
            
            if trans_dim == 1:
                # Original method: single heatmap
                trans_point = out["trans"][:, :, 0, :, :]  # (bs, nc, h, w)
                q_trans = trans_point.view(bs, nc, h * w).transpose(1, 2)  # (bs, h*w, nc)
                if not only_pred:
                    q_trans = q_trans.clone()
                
                if self.stage_two:
                    out = out["mvt2"]
                    trans_point2 = out["trans"][:, :, 0, :, :]
                    q_trans2 = trans_point2.view(bs, nc, h * w).transpose(1, 2)
                    if not only_pred:
                        q_trans2 = q_trans2.clone()
                    q_trans = torch.cat((q_trans, q_trans2), dim=2)
            
            else:
                # 4-point method: 4 heatmaps (one for each point)
                # We need to return all 4, stacked
                # out["trans"]: (bs, nc, 4, h, w)
                q_trans_list = []
                for i in range(trans_dim):
                    trans_point = out["trans"][:, :, i, :, :]  # (bs, nc, h, w)
                    q_trans_i = trans_point.view(bs, nc, h * w).transpose(1, 2)  # (bs, h*w, nc)
                    if not only_pred:
                        q_trans_i = q_trans_i.clone()
                    q_trans_list.append(q_trans_i)
                
                # Stack: (bs, h*w, nc, 4)
                q_trans = torch.stack(q_trans_list, dim=-1)
                
                if self.stage_two:
                    out = out["mvt2"]
                    q_trans_list2 = []
                    for i in range(trans_dim):
                        trans_point2 = out["trans"][:, :, i, :, :]
                        q_trans2_i = trans_point2.view(bs, nc, h * w).transpose(1, 2)
                        if not only_pred:
                            q_trans2_i = q_trans2_i.clone()
                        q_trans_list2.append(q_trans2_i)
                    q_trans2 = torch.stack(q_trans_list2, dim=-1)
                    # Concatenate along camera dimension: (bs, h*w, nc*2, 4)
                    q_trans = torch.cat((q_trans, q_trans2), dim=2)
        else:
            pts = None
            q_trans = None
            if self.stage_two:
                out = out["mvt2"]

        if self.rot_ver == 0:
            # (bs, 218)
            rot_q = out["feat"].view(bs, -1)[:, 0 : self.num_all_rot] # b, 216
            grip_q = out["feat"].view(bs, -1)[:, self.num_all_rot : self.num_all_rot + 2] # b, 2
            # (bs, 2)
            collision_q = out["feat"].view(bs, -1)[
                :, self.num_all_rot + 2 : self.num_all_rot + 4
            ] # b, 2
        elif self.rot_ver == 1:
            rot_q = torch.cat((out["feat_x"], out["feat_y"], out["feat_z"]),
                              dim=-1).view(bs, -1)
            grip_q = out["feat_ex_rot"].view(bs, -1)[:, :2]
            collision_q = out["feat_ex_rot"].view(bs, -1)[:, 2:]
        else:
            assert False

        y_q = None

        return q_trans, rot_q, grip_q, collision_q, y_q, pts

    def update(
        self,
        step: int,
        replay_sample: dict,
        backprop: bool = True,
        eval_log: bool = False,
        reset_log: bool = False,
    ) -> dict:
        assert replay_sample["rot_grip_action_indicies"].shape[1:] == (1, 4)
        assert replay_sample["ignore_collisions"].shape[1:] == (1, 1)
        assert replay_sample["gripper_pose"].shape[1:] == (1, 7)
        assert replay_sample["lang_goal_embs"].shape[1:] == (1, 77, 512)
        assert replay_sample["low_dim_state"].shape[1:] == (
            1,
            self._net_mod.proprio_dim,
        )

        # sample
        action_rot_grip = replay_sample["rot_grip_action_indicies"][
            :, -1
        ].int()  # (b, 4) of int
        action_ignore_collisions = replay_sample["ignore_collisions"][
            :, -1
        ].int()  # (b, 1) of int
        action_gripper_pose = replay_sample["gripper_pose"][:, -1]  # (b, 7)
        action_trans_con = action_gripper_pose[:, 0:3]  # (b, 3)
        # rotation in quaternion xyzw
        action_rot = action_gripper_pose[:, 3:7]  # (b, 4)
        action_grip = action_rot_grip[:, -1]  # (b,)
        
        lang_goal_embs = replay_sample["lang_goal_embs"][:, -1].float()
        tasks = replay_sample["tasks"]

        proprio = arm_utils.stack_on_channel(replay_sample["low_dim_state"])  # (b, 4)
        return_out = {}

        obs, pcd = peract_utils._preprocess_inputs(replay_sample, self.cameras)

        with torch.no_grad():
            pc, img_feat = rvt_utils.get_pc_img_feat(
                obs,
                pcd,
            )

            if self._transform_augmentation and backprop:
                action_trans_con, action_rot, pc = apply_se3_aug_con(
                    pcd=pc,
                    action_gripper_pose=action_gripper_pose,
                    bounds=torch.tensor(self.scene_bounds),
                    trans_aug_range=torch.tensor(self._transform_augmentation_xyz),
                    rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                )
                action_trans_con = torch.tensor(action_trans_con).to(pc.device)
                action_rot = torch.tensor(action_rot).to(pc.device)

            # TODO: vectorize
            action_rot = action_rot.cpu().numpy()
            for i, _action_rot in enumerate(action_rot):
                _action_rot = aug_utils.normalize_quaternion(_action_rot)
                if _action_rot[-1] < 0:
                    _action_rot = -_action_rot
                action_rot[i] = _action_rot

            pc, img_feat = rvt_utils.move_pc_in_bound(
                pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
            )
            wpt = [x[:3] for x in action_trans_con]

            wpt_local = []
            rev_trans = []
            for _pc, _wpt in zip(pc, wpt):
                a, b = mvt_utils.place_pc_in_cube(
                    _pc,
                    _wpt,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )
                wpt_local.append(a.unsqueeze(0))
                rev_trans.append(b)

            wpt_local = torch.cat(wpt_local, axis=0)
            
            # Create ground truth 4 points in LOCAL coordinates (after transformation)
            gt_points4 = None
            if self.use_4point_pose:
                # Convert quaternion to rotation matrix
                action_rot_tensor = torch.tensor(action_rot, device=wpt_local.device).float()
                action_rot_matrix = torch3d_tf.quaternion_to_matrix(action_rot_tensor)
                # Convert gripper state to boolean
                action_grip_bool = action_grip.bool()
                
                # Generate 4 points from ground truth pose IN LOCAL COORDINATES
                # Need to transform each ground truth point to local coordinates
                gt_points4_list = []
                for i, (_action_trans, _action_rot_matrix, _action_grip_bool, _rev_trans) in enumerate(
                    zip(action_trans_con, action_rot_matrix, action_grip_bool, rev_trans)
                ):
                    # Generate 4 points in world coordinates
                    gt_p0_world, gt_p1_world, gt_p2_world, gt_p3_world = mvt_utils.pose_to_points4(
                        _action_trans.unsqueeze(0), _action_rot_matrix.unsqueeze(0), _action_grip_bool.unsqueeze(0)
                    )
                    
                    # Transform each point to local coordinates using the inverse transform
                    # The rev_trans function transforms from local to world, so we need to find local coords
                    # that would map to these world coords
                    # For simplicity, we can use place_pc_in_cube on each point
                    gt_p0_local, _ = mvt_utils.place_pc_in_cube(
                        pc[i], gt_p0_world.squeeze(0),
                        with_mean_or_bounds=self._place_with_mean,
                        scene_bounds=None if self._place_with_mean else self.scene_bounds,
                    )
                    gt_p1_local, _ = mvt_utils.place_pc_in_cube(
                        pc[i], gt_p1_world.squeeze(0),
                        with_mean_or_bounds=self._place_with_mean,
                        scene_bounds=None if self._place_with_mean else self.scene_bounds,
                    )
                    gt_p2_local, _ = mvt_utils.place_pc_in_cube(
                        pc[i], gt_p2_world.squeeze(0),
                        with_mean_or_bounds=self._place_with_mean,
                        scene_bounds=None if self._place_with_mean else self.scene_bounds,
                    )
                    gt_p3_local, _ = mvt_utils.place_pc_in_cube(
                        pc[i], gt_p3_world.squeeze(0),
                        with_mean_or_bounds=self._place_with_mean,
                        scene_bounds=None if self._place_with_mean else self.scene_bounds,
                    )
                    
                    # Stack the 4 points for this sample
                    gt_4points_sample = torch.stack([gt_p0_local, gt_p1_local, gt_p2_local, gt_p3_local], dim=0)
                    gt_points4_list.append(gt_4points_sample.unsqueeze(0))
                
                gt_points4 = torch.cat(gt_points4_list, dim=0)  # (bs, 4, 3)

            # TODO: Vectorize
            pc = [
                mvt_utils.place_pc_in_cube(
                    _pc,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )[0]
                for _pc in pc
            ]

            bs = len(pc)
            nc = self._net_mod.num_img
            h = w = self._net_mod.img_size

            if backprop and (self.img_aug != 0):
                img_aug = self.img_aug
            else:
                img_aug = 0

            dyn_cam_info = None

        with autocast(enabled=self.amp):
            (
                action_rot_x_one_hot,
                action_rot_y_one_hot,
                action_rot_z_one_hot,
                action_grip_one_hot,  # (bs, 2)
                action_collision_one_hot,  # (bs, 2)
            ) = self._get_one_hot_expert_actions(
                bs, action_rot, action_grip, action_ignore_collisions, device=self._device
            )

            if self.rot_ver == 1:
                rot_x_y = torch.cat(
                    [
                        action_rot_x_one_hot.argmax(dim=-1, keepdim=True),
                        action_rot_y_one_hot.argmax(dim=-1, keepdim=True),
                    ],
                    dim=-1,
                )
                if self.rot_x_y_aug != 0:
                    # add random interger between -rot_x_y_aug and rot_x_y_aug to rot_x_y
                    rot_x_y += torch.randint(
                        -self.rot_x_y_aug, self.rot_x_y_aug, size=rot_x_y.shape
                    ).to(rot_x_y.device)
                    rot_x_y %= self._num_rotation_classes

            out = self._network(
                pc=pc,
                img_feat=img_feat,
                proprio=proprio,
                lang_emb=lang_goal_embs,
                img_aug=img_aug,
                wpt_local=wpt_local if self._network.training else None,
                rot_x_y=rot_x_y if self.rot_ver == 1 else None,
            )

            q_trans, rot_q, grip_q, collision_q, y_q, pts = self.get_q(
                out, dims=(bs, nc, h, w)
            )

            # Create ground truth heatmaps (pass gt_points4 if using 4-point pose)
            if self.use_4point_pose:
                action_trans = self.get_action_trans(
                    wpt_local, pts, out, dyn_cam_info, dims=(bs, nc, h, w),
                    gt_points4_local=gt_points4
                )
            else:
                action_trans = self.get_action_trans(
                    wpt_local, pts, out, dyn_cam_info, dims=(bs, nc, h, w)
                )

        loss_log = {}
        if backprop:
            with autocast(enabled=self.amp):
                if self.use_4point_pose:
                    # 4-point loss calculation using cross-entropy on heatmaps
                    # q_trans shape: (bs, h*w, nc, 4) for 4-point method
                    # action_trans shape: (bs, h*w, nc, 4) for 4-point method
                    
                    # Compute cross-entropy loss for each of the 4 points
                    # Extract predictions for each point from q_trans
                    trans_loss_p0 = self._cross_entropy_loss(q_trans[..., 0], action_trans[..., 0]).mean()
                    trans_loss_p1 = self._cross_entropy_loss(q_trans[..., 1], action_trans[..., 1]).mean()
                    trans_loss_p2 = self._cross_entropy_loss(q_trans[..., 2], action_trans[..., 2]).mean()
                    trans_loss_p3 = self._cross_entropy_loss(q_trans[..., 3], action_trans[..., 3]).mean()
                    
                    # Total translation loss is sum of all 4 points
                    trans_loss = trans_loss_p0 + trans_loss_p1 + trans_loss_p2 + trans_loss_p3
                    
                    # Collision loss (same as original)
                    collision_loss = self._cross_entropy_loss(
                        collision_q, action_collision_one_hot.argmax(-1)
                    ).mean()
                    
                    total_loss = trans_loss + collision_loss
                    
                    # Update loss log for 4-point method
                    loss_log = {
                        "total_loss": total_loss.item(),
                        "trans_loss": trans_loss.item(),
                        "trans_loss_p0": trans_loss_p0.item(),
                        "trans_loss_p1": trans_loss_p1.item(),
                        "trans_loss_p2": trans_loss_p2.item(),
                        "trans_loss_p3": trans_loss_p3.item(),
                        "collision_loss": collision_loss.item(),
                        "lr": self._optimizer.param_groups[0]["lr"],
                    }
                    
                else:
                    # Original loss calculation
                    # cross-entropy loss
                    trans_loss = self._cross_entropy_loss(q_trans, action_trans).mean()
                    rot_loss_x = rot_loss_y = rot_loss_z = 0.0
                    grip_loss = 0.0
                    collision_loss = 0.0
                    if self.add_rgc_loss:
                        rot_loss_x = self._cross_entropy_loss(
                            rot_q[
                                :,
                                0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                            ],
                            action_rot_x_one_hot.argmax(-1),
                        ).mean()

                        rot_loss_y = self._cross_entropy_loss(
                            rot_q[
                                :,
                                1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                            ],
                            action_rot_y_one_hot.argmax(-1),
                        ).mean()

                        rot_loss_z = self._cross_entropy_loss(
                            rot_q[
                                :,
                                2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                            ],
                            action_rot_z_one_hot.argmax(-1),
                        ).mean()

                        grip_loss = self._cross_entropy_loss(
                            grip_q,
                            action_grip_one_hot.argmax(-1),
                        ).mean()

                        collision_loss = self._cross_entropy_loss(
                            collision_q, action_collision_one_hot.argmax(-1)
                        ).mean()

                    total_loss = (
                        trans_loss
                        + rot_loss_x
                        + rot_loss_y
                        + rot_loss_z
                        + grip_loss
                        + collision_loss
                    )
                    
                    # Update loss log for original method
                    loss_log = {
                        "total_loss": total_loss.item(),
                        "trans_loss": trans_loss.item(),
                        "rot_loss_x": rot_loss_x.item(),
                        "rot_loss_y": rot_loss_y.item(),
                        "rot_loss_z": rot_loss_z.item(),
                        "grip_loss": grip_loss.item(),
                        "collision_loss": collision_loss.item(),
                        "lr": self._optimizer.param_groups[0]["lr"],
                    }

            self._optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self._optimizer)
            self.scaler.update()
            self._lr_sched.step()

            manage_loss_log(self, loss_log, reset_log=reset_log)
            return_out.update(loss_log)

        if eval_log:
            with torch.no_grad():
                wpt = torch.cat([x.unsqueeze(0) for x in wpt])
                pred_wpt, pred_rot_quat, _, _ = self.get_pred(
                    out,
                    rot_q,
                    grip_q,
                    collision_q,
                    y_q,
                    rev_trans,
                    dyn_cam_info=dyn_cam_info,
                )

                return_log = manage_eval_log(
                    self=self,
                    tasks=tasks,
                    wpt=wpt,
                    pred_wpt=pred_wpt,
                    action_rot=action_rot,
                    pred_rot_quat=pred_rot_quat,
                    action_grip_one_hot=action_grip_one_hot,
                    grip_q=grip_q,
                    action_collision_one_hot=action_collision_one_hot,
                    collision_q=collision_q,
                    reset_log=reset_log,
                )

                return_out.update(return_log)

        return return_out

    @torch.no_grad()
    def act(
        self, step: int, observation: dict, deterministic=True, pred_distri=False
    ) -> ActResult:
        if self.add_lang:
            lang_goal_tokens = observation.get("lang_goal_tokens", None).long()
            _, lang_goal_embs = _clip_encode_text(self.clip_model, lang_goal_tokens[0])
            lang_goal_embs = lang_goal_embs.float()
        else:
            lang_goal_embs = (
                torch.zeros(observation["lang_goal_embs"].shape)
                .float()
                .to(self._device)
            )

        proprio = arm_utils.stack_on_channel(observation["low_dim_state"])

        obs, pcd = peract_utils._preprocess_inputs(observation, self.cameras)
        pc, img_feat = rvt_utils.get_pc_img_feat(
            obs,
            pcd,
        )

        pc, img_feat = rvt_utils.move_pc_in_bound(
            pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
        )

        # TODO: Vectorize
        pc_new = []
        rev_trans = []
        for _pc in pc:
            a, b = mvt_utils.place_pc_in_cube(
                _pc,
                with_mean_or_bounds=self._place_with_mean,
                scene_bounds=None if self._place_with_mean else self.scene_bounds,
            )
            pc_new.append(a)
            rev_trans.append(b)
        pc = pc_new

        bs = len(pc)
        nc = self._net_mod.num_img
        h = w = self._net_mod.img_size
        dyn_cam_info = None

        out = self._network(
            pc=pc,
            img_feat=img_feat,
            proprio=proprio,
            lang_emb=lang_goal_embs,
            img_aug=0,  # no img augmentation while acting
        )
        _, rot_q, grip_q, collision_q, y_q, _ = self.get_q(
            out, dims=(bs, nc, h, w), only_pred=True, get_q_trans=False
        )
        pred_wpt, pred_rot_quat, pred_grip, pred_coll = self.get_pred(
            out, rot_q, grip_q, collision_q, y_q, rev_trans, dyn_cam_info
        )

        continuous_action = np.concatenate(
            (
                pred_wpt[0, 0, :].cpu().numpy(),  # Use first waypoint from first batch item
                pred_rot_quat[0],
                pred_grip[0].cpu().numpy(),
                pred_coll[0].cpu().numpy(),
            )
        )
        if pred_distri:
            x_distri = rot_q[
                0,
                0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
            ]
            y_distri = rot_q[
                0,
                1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
            ]
            z_distri = rot_q[
                0,
                2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
            ]
            return ActResult(continuous_action), (
                x_distri.cpu().numpy(),
                y_distri.cpu().numpy(),
                z_distri.cpu().numpy(),
            )
        else:
            return ActResult(continuous_action)

    def get_pred(
        self,
        out,
        rot_q,
        grip_q,
        collision_q,
        y_q,
        rev_trans,
        dyn_cam_info,
    ):
        if self.stage_two:
            assert y_q is None
            mvt1_or_mvt2 = False
        else:
            mvt1_or_mvt2 = True

        pred_wpt_local = self._net_mod.get_wpt(
            out, mvt1_or_mvt2, dyn_cam_info, y_q
        )

        # Process waypoints - handle both 1 point and 4 points cases
        pred_wpt = []
        num_points = pred_wpt_local.shape[1]  # 1 for original method, 4 for 4-point method
        for i in range(num_points):
            pred_wpt_point = []
            for _pred_wpt_local, _rev_trans in zip(pred_wpt_local[:, i, :], rev_trans):
                pred_wpt_point.append(_rev_trans(_pred_wpt_local))
            pred_wpt_point = torch.cat([x.unsqueeze(0) for x in pred_wpt_point])
            pred_wpt.append(pred_wpt_point)
        pred_wpt = torch.stack(pred_wpt, dim=1)  # (bs, num_points, 3)

        if self.use_4point_pose:
            # Ensure we have 4 points for 4-point pose method
            assert pred_wpt.shape[1] == 4, f"Expected 4 points for 4-point pose, got {pred_wpt.shape[1]}"
            
            # Extract the 4 points: p0, p1, p2, p3
            p0 = pred_wpt[:, 0, :]  # Base point
            p1 = pred_wpt[:, 1, :]  # Left finger
            p2 = pred_wpt[:, 2, :]  # Right finger
            p3 = pred_wpt[:, 3, :]  # Approach direction
            
            # Calculate pose using the 4 points (converts to pose representation)
            trans, rot_matrix, is_open = mvt_utils.points4_to_pose(
                p0, p1, p2, p3, 
                to_6d=False,  # Get rotation matrix
                align_z_to_p3=True
            )
            
            # Now we have trans, rot, gripper - same as original method would have
            # Create pred_wpt in same format as original method: (bs, 1, 3) with center translation
            pred_wpt = trans.unsqueeze(1)  # (bs, 1, 3)
            
            # Convert rotation matrix to quaternion (same format as original method)
            pred_rot_quat = torch3d_tf.matrix_to_quaternion(rot_matrix)
            pred_rot_quat = pred_rot_quat.cpu().numpy().astype('float64')
            
            # Convert gripper boolean to int format (same as original method)
            pred_grip = torch.zeros((trans.shape[0], 1), dtype=torch.int64, device=trans.device)
            pred_grip[is_open, 0] = 1
            
        else:
            pred_rot = torch.cat(
                (
                    rot_q[
                        :,
                        0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                    ].argmax(1, keepdim=True),
                    rot_q[
                        :,
                        1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                    ].argmax(1, keepdim=True),
                    rot_q[
                        :,
                        2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                    ].argmax(1, keepdim=True),
                ),
                dim=-1,
            )
            pred_rot_quat = aug_utils.discrete_euler_to_quaternion(
                pred_rot.cpu(), self._rotation_resolution
            )
            pred_grip = grip_q.argmax(1, keepdim=True)

        # Keep collision prediction the same for both methods
        pred_coll = collision_q.argmax(1, keepdim=True)

        return pred_wpt, pred_rot_quat, pred_grip, pred_coll

    @torch.no_grad()
    def get_action_trans(
        self,
        wpt_local,
        pts,
        out,
        dyn_cam_info,
        dims,
        gt_points4_local=None,
    ):
        """
        Generate ground truth heatmaps for translation points.
        For 4-point pose: generates 4 separate heatmaps
        For original: generates 1 heatmap
        """
        bs, nc, h, w = dims
        
        if self.use_4point_pose and gt_points4_local is not None:
            # Generate heatmaps for all 4 points
            # gt_points4_local: (bs, 4, 3) in local coordinates
            action_trans_list = []
            
            for point_idx in range(4):  # For each of the 4 points
                # Get the i-th point for all samples: (bs, 3)
                wpt_i = gt_points4_local[:, point_idx, :]
                
                # Project to image coordinates
                wpt_img = self._net_mod.get_pt_loc_on_img(
                    wpt_i.unsqueeze(1),  # (bs, 1, 3)
                    mvt1_or_mvt2=True,
                    dyn_cam_info=dyn_cam_info,
                    out=None
                )
                assert wpt_img.shape[1] == 1
                
                if self.stage_two:
                    wpt_img2 = self._net_mod.get_pt_loc_on_img(
                        wpt_i.unsqueeze(1),
                        mvt1_or_mvt2=False,
                        dyn_cam_info=dyn_cam_info,
                        out=out,
                    )
                    assert wpt_img2.shape[1] == 1
                    wpt_img = torch.cat((wpt_img, wpt_img2), dim=-2)
                
                # (bs, num_img, 2)
                wpt_img = wpt_img.squeeze(1)
                
                # Generate heatmap for this point
                action_trans_i = mvt_utils.generate_hm_from_pt(
                    wpt_img.reshape(-1, 2),
                    (h, w),
                    sigma=self.gt_hm_sigma,
                    thres_sigma_times=3,
                )
                nc_actual = nc * 2 if self.stage_two else nc
                action_trans_i = action_trans_i.view(bs, nc_actual, h * w).transpose(1, 2).clone()
                action_trans_list.append(action_trans_i)
            
            # Stack all 4 heatmaps: list of 4 × (bs, h*w, nc) -> (bs, h*w, nc, 4)
            action_trans = torch.stack(action_trans_list, dim=-1)  # (bs, h*w, nc, 4)
            
        else:
            # Original method: single point heatmap
            wpt_img = self._net_mod.get_pt_loc_on_img(
                wpt_local.unsqueeze(1),
                mvt1_or_mvt2=True,
                dyn_cam_info=dyn_cam_info,
                out=None
            )
            assert wpt_img.shape[1] == 1
            if self.stage_two:
                wpt_img2 = self._net_mod.get_pt_loc_on_img(
                    wpt_local.unsqueeze(1),
                    mvt1_or_mvt2=False,
                    dyn_cam_info=dyn_cam_info,
                    out=out,
                )
                assert wpt_img2.shape[1] == 1
                wpt_img = torch.cat((wpt_img, wpt_img2), dim=-2)
                nc = nc * 2

            # (bs, num_img, 2)
            wpt_img = wpt_img.squeeze(1)

            action_trans = mvt_utils.generate_hm_from_pt(
                wpt_img.reshape(-1, 2),
                (h, w),
                sigma=self.gt_hm_sigma,
                thres_sigma_times=3,
            )
            action_trans = action_trans.view(bs, nc, h * w).transpose(1, 2).clone()

        return action_trans

    def reset(self):
        pass

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()
