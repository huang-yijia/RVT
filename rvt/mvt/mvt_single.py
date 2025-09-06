# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from math import ceil

import torch
import torch.nn.functional as F

from torch import nn
from einops import rearrange, repeat

import rvt.mvt.utils as mvt_utils
from rvt.mvt.attn import (
    Conv2DBlock,
    Conv2DUpsampleBlock,
    PreNorm,
    Attention,
    cache_fn,
    DenseBlock,
    FeedForward,
    FixedPositionalEncoding,
)
from rvt.mvt.raft_utils import ConvexUpSample



class MVT(nn.Module):
    def __init__(
        self,
        depth, # 8
        img_size, # 220
        add_proprio, # True
        proprio_dim, # 4
        add_lang, # True
        lang_dim, # 512
        lang_len, # 77
        img_feat_dim, # 3
        feat_dim, # (72 * 3) + 2 + 2 = 220
        im_channels, # 64
        attn_dim, # 512
        attn_heads, # 8
        attn_dim_head, # 64
        activation, # lrelu
        weight_tie_layers, # False
        attn_dropout, # 0.1
        decoder_dropout, # 0.0
        img_patch_size, # 11
        final_dim, # 64
        trans_dim, # 1
        self_cross_ver, # 1
        add_corr, # True
        norm_corr, # False
        add_pixel_loc, # True
        add_depth, # True
        rend_three_views, # False
        use_point_renderer, # False
        pe_fix, # True
        feat_ver, # 0
        wpt_img_aug, # 0.01
        inp_pre_pro, # True
        inp_pre_con, # True
        cvx_up, # False
        xops, # False
        rot_ver, # 0
        num_rot, # 72
        renderer_device="cuda:0",
        renderer=None,
        no_feat=False,
    ):
        """MultiView Transfomer

        :param depth: depth of the attention network
        :param img_size: number of pixels per side for rendering
        :param renderer_device: device for placing the renderer
        :param add_proprio:
        :param proprio_dim:
        :param add_lang:
        :param lang_dim:
        :param lang_len:
        :param img_feat_dim:
        :param feat_dim:
        :param im_channels: intermediate channel size
        :param attn_dim:
        :param attn_heads:
        :param attn_dim_head:
        :param activation:
        :param weight_tie_layers:
        :param attn_dropout:
        :param decoder_dropout:
        :param img_patch_size: intial patch size
        :param final_dim: final dimensions of features
        :param trans_dim: dimensions of the translation decoder
        :param self_cross_ver:
        :param add_corr:
        :param norm_corr: wether or not to normalize the correspondece values.
            this matters when pc is outide -1, 1 like for the two stage mvt
        :param add_pixel_loc:
        :param add_depth:
        :param rend_three_views: True/False. Render only three views,
            i.e. top, right and front. Overwrites other configurations.
        :param use_point_renderer: whether to use the point renderer or not
        :param pe_fix: matter only when add_lang is True
            Either:
                True: use position embedding only for image topkens
                False: use position embedding for lang and image token
        :param feat_ver: whether to max pool final features or use soft max
            values using the heamtmap
        :param wpt_img_aug: how much noise is added to the wpt_img while
            training, expressed as a percentage of the image size
        :param inp_pre_pro: whether or not we have the intial input
            preprocess layer. this was used in peract but not not having it has
            cost advantages. if false, we replace the 1D convolution in the
            orginal design with identity
        :param inp_pre_con: whether or not the output of the inital
            preprocessing layer is concatenated with the ouput of the
            upsampling layer for the "final" layer
        :param cvx_up: whether to use learned convex upsampling
        :param xops: whether to use xops or not
        :param rot_ver: version of the rotation prediction network
            Either:
                0: same as peract, independent discrete xyz predictions
                1: xyz prediction dependent on one another
        :param num_rot: number of discrete rotations per axis, used only when
            rot_ver is 1
        :param no_feat: whether to return features or not
        """

        super().__init__()
        self.depth = depth # 8
        self.img_feat_dim = img_feat_dim # 3
        self.img_size = img_size # 220
        self.add_proprio = add_proprio # True
        self.proprio_dim = proprio_dim # 4
        self.add_lang = add_lang # True
        self.lang_dim = lang_dim # 512
        self.lang_len = lang_len # 77 
        self.im_channels = im_channels # 64
        self.img_patch_size = img_patch_size # 11
        self.final_dim = final_dim # 64
        self.trans_dim = trans_dim # 1
        self.attn_dropout = attn_dropout # 0.1
        self.decoder_dropout = decoder_dropout # 0.0
        self.self_cross_ver = self_cross_ver # 1
        self.add_corr = add_corr # True
        self.norm_corr = norm_corr # False
        self.add_pixel_loc = add_pixel_loc # True
        self.add_depth = add_depth # True
        self.pe_fix = pe_fix # True
        self.feat_ver = feat_ver # 0
        self.wpt_img_aug = wpt_img_aug # 0.01
        self.inp_pre_pro = inp_pre_pro # True
        self.inp_pre_con = inp_pre_con # True
        self.cvx_up = cvx_up # False
        self.use_point_renderer = use_point_renderer # False
        self.rot_ver = rot_ver # 0
        self.num_rot = num_rot # 72
        self.no_feat = no_feat # False

        if self.cvx_up:
            assert not self.inp_pre_con, (
                "When using the convex upsampling, we do not concatenate"
                " features from input_preprocess to the features used for"
                " prediction"
            )

        print(f"MVT Vars: {vars(self)}")

        assert not renderer is None
        self.renderer = renderer
        self.num_img = self.renderer.num_img

        # patchified input dimensions
        spatial_size = img_size // self.img_patch_size  # 220 / 11  = 20

        if self.add_proprio:
            # 64 img features + 64 proprio features
            self.input_dim_before_seq = self.im_channels * 2 # 64 * 2 = 128
        else:
            self.input_dim_before_seq = self.im_channels 

        # learnable positional encoding
        if add_lang:
            lang_emb_dim, lang_max_seq_len = lang_dim, lang_len # 512, 77
        else:
            lang_emb_dim, lang_max_seq_len = 0, 0
        self.lang_emb_dim = lang_emb_dim # 512
        self.lang_max_seq_len = lang_max_seq_len # 77

        if self.pe_fix:
            num_pe_token = spatial_size**2 * self.num_img # 20**2 * 5 = 2000
        else:
            num_pe_token = lang_max_seq_len + (spatial_size**2 * self.num_img)
        self.pos_encoding = nn.Parameter(
            torch.randn(
                1,
                num_pe_token, # 2000
                self.input_dim_before_seq, # 128
            )
        )

        inp_img_feat_dim = self.img_feat_dim # 3
        if self.add_corr:
            inp_img_feat_dim += 3
        if self.add_pixel_loc:
            inp_img_feat_dim += 3
            self.pixel_loc = torch.zeros(
                (self.num_img, 3, self.img_size, self.img_size)
            )
            self.pixel_loc[:, 0, :, :] = (
                torch.linspace(-1, 1, self.num_img).unsqueeze(-1).unsqueeze(-1)
            )
            self.pixel_loc[:, 1, :, :] = (
                torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(-1)
            )
            self.pixel_loc[:, 2, :, :] = (
                torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(0)
            )
        if self.add_depth:
            inp_img_feat_dim += 1

        # img input preprocessing encoder
        if self.inp_pre_pro:
            self.input_preprocess = Conv2DBlock(
                inp_img_feat_dim, # 7
                self.im_channels, # 64
                kernel_sizes=1,
                strides=1,
                norm=None,
                activation=activation, #lrelu
            )
            inp_pre_out_dim = self.im_channels # 64
        else:
            # identity
            self.input_preprocess = lambda x: x
            inp_pre_out_dim = inp_img_feat_dim

        if self.add_proprio:
            # proprio preprocessing encoder
            self.proprio_preprocess = DenseBlock(
                self.proprio_dim, # 4
                self.im_channels, # 64
                norm="group",
                activation=activation, # lrelu
            )

        self.patchify = Conv2DBlock(
            inp_pre_out_dim, # 64
            self.im_channels, # 64
            kernel_sizes=self.img_patch_size, # 11
            strides=self.img_patch_size, # 11
            norm="group",
            activation=activation,
            padding=0,
        )

        # lang preprocess
        if self.add_lang:
            self.lang_preprocess = DenseBlock(
                lang_emb_dim, # 512
                self.im_channels * 2, # 128
                norm="group",
                activation=activation,
            )

        self.fc_bef_attn = DenseBlock(
            self.input_dim_before_seq, # 128
            attn_dim, # 512
            norm=None,
            activation=None,
        )
        self.fc_aft_attn = DenseBlock(
            attn_dim, # 512
            self.input_dim_before_seq, # 128
            norm=None,
            activation=None,
        )

        get_attn_attn = lambda: PreNorm(
            attn_dim, # 512
            Attention(
                attn_dim, # 512
                heads=attn_heads, # 8
                dim_head=attn_dim_head, # 64
                dropout=attn_dropout, # 0.1
                use_fast=xops, # False
            ),
        )
        get_attn_ff = lambda: PreNorm(attn_dim, FeedForward(attn_dim))
        get_attn_attn, get_attn_ff = map(cache_fn, (get_attn_attn, get_attn_ff))
        # self-attention layers
        self.layers = nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}
        attn_depth = depth

        # 8 self-attention layers
        for _ in range(attn_depth):
            self.layers.append(
                nn.ModuleList([get_attn_attn(**cache_args), get_attn_ff(**cache_args)]) 
            )

        if cvx_up:
            self.up0 = ConvexUpSample(
                in_dim=self.input_dim_before_seq,
                out_dim=1,
                up_ratio=self.img_patch_size,
            )
        else:
            self.up0 = Conv2DUpsampleBlock(
                self.input_dim_before_seq, # 128
                self.im_channels, # 64
                kernel_sizes=self.img_patch_size, # 11
                strides=self.img_patch_size, # 11
                norm=None,
                activation=activation,
                out_size=self.img_size, # 220
            )

            if self.inp_pre_con:
                final_inp_dim = self.im_channels + inp_pre_out_dim # 128
            else:
                final_inp_dim = self.im_channels

            # final layers
            self.final = Conv2DBlock(
                final_inp_dim, # 128
                self.im_channels, # 64
                kernel_sizes=3,
                strides=1,
                norm=None,
                activation=activation,
            )

            # 3D translation decoder (Heatmap)
            self.trans_decoder = Conv2DBlock(
                self.final_dim, # 64
                self.trans_dim, # 1
                kernel_sizes=3,
                strides=1,
                norm=None,
                activation=None,
            )

        if not self.no_feat:
            feat_fc_dim = 0
            feat_fc_dim += self.input_dim_before_seq # 128
            if self.cvx_up:
                feat_fc_dim += self.input_dim_before_seq
            else:
                feat_fc_dim += self.final_dim # 192

            def get_feat_fc(
                _feat_in_size, # 960
                _feat_out_size, # 220
                _feat_fc_dim=feat_fc_dim, # 192
            ):
                """
                _feat_in_size: input feature size
                _feat_out_size: output feature size
                _feat_fc_dim: hidden feature size
                """
                layers = [
                    nn.Linear(_feat_in_size, _feat_fc_dim), # 960 -> 192
                    nn.ReLU(),
                    nn.Linear(_feat_fc_dim, _feat_fc_dim // 2), # 192 -> 96
                    nn.ReLU(),
                    nn.Linear(_feat_fc_dim // 2, _feat_out_size), # 96 -> 220, 96 -> 4
                ]
                feat_fc = nn.Sequential(*layers)
                return feat_fc

            feat_out_size = feat_dim # (72 * 3) + 2 + 2 = 220

            if self.rot_ver == 0:
                self.feat_fc = get_feat_fc(
                    self.num_img * feat_fc_dim, # 5 * 192 = 960
                    feat_out_size, # 220
                ) # b, 220
            elif self.rot_ver == 1:
                assert self.num_rot * 3 <= feat_out_size
                feat_out_size_ex_rot = feat_out_size - (self.num_rot * 3) # 220 - (72 * 3) = 4
                if feat_out_size_ex_rot > 0:
                    self.feat_fc_ex_rot = get_feat_fc(
                        self.num_img * feat_fc_dim, # 960
                        feat_out_size_ex_rot, # 4
                    ) # b, 4

                self.feat_fc_init_bn = nn.BatchNorm1d(self.num_img * feat_fc_dim) # b, 960
                self.feat_fc_pe = FixedPositionalEncoding(
                    self.num_img * feat_fc_dim, feat_scale_factor=1
                )
                self.feat_fc_x = get_feat_fc(self.num_img * feat_fc_dim, self.num_rot) # b, 72
                self.feat_fc_y = get_feat_fc(self.num_img * feat_fc_dim, self.num_rot) # b, 72
                self.feat_fc_z = get_feat_fc(self.num_img * feat_fc_dim, self.num_rot) # b, 72

            else:
                assert False

        if self.use_point_renderer:
            from point_renderer.rvt_ops import select_feat_from_hm
        else:
            from mvt.renderer import select_feat_from_hm
        global select_feat_from_hm

    def get_pt_loc_on_img(self, pt, dyn_cam_info):
        """
        transform location of points in the local frame to location on the
        image
        :param pt: (bs, np, 3)
        :return: pt_img of size (bs, np, num_img, 2)
        """
        pt_img = self.renderer.get_pt_loc_on_img(
            pt, fix_cam=True, dyn_cam_info=dyn_cam_info
        )
        return pt_img

    def forward(
        self,
        img,
        proprio=None,
        lang_emb=None,
        wpt_local=None,
        rot_x_y=None,
        **kwargs,
    ):
        """
        :param img: tensor of shape (bs, num_img, img_feat_dim, h, w)
        :param proprio: tensor of shape (bs, priprio_dim)
        :param lang_emb: tensor of shape (bs, lang_len, lang_dim)
        :param img_aug: (float) magnitude of augmentation in rgb image
        :param rot_x_y: (bs, 2)
        """

        bs, num_img, img_feat_dim, h, w = img.shape # b, 5, 3, 220, 220
        num_pat_img = h // self.img_patch_size # 220 / 11 = 20
        assert num_img == self.num_img # 5
        # assert img_feat_dim == self.img_feat_dim
        assert h == w == self.img_size # 220

        img = img.view(bs * num_img, img_feat_dim, h, w)
        # preprocess
        # (bs * num_img, im_channels, h, w)
        d0 = self.input_preprocess(img) # b, 5, 64, 220, 220

        # (bs * num_img, im_channels, h, w) ->
        # (bs * num_img, im_channels, h / img_patch_strid, w / img_patch_strid) patches
        ins = self.patchify(d0) # b, 5, 64, 20, 20
        # (bs, im_channels, num_img, h / img_patch_strid, w / img_patch_strid) patches
        ins = (
            ins.view(
                bs,
                num_img,
                self.im_channels,
                num_pat_img,
                num_pat_img,
            )
            .transpose(1, 2) # b, 64, 5, 20, 20
            .clone()
        )

        # concat proprio
        _, _, _d, _h, _w = ins.shape
        if self.add_proprio:
            p = self.proprio_preprocess(proprio)  # [B,4] -> [B,64]
            p = p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, _d, _h, _w) # [B,64,1,1,1] -> [B,64,5,20,20]
            ins = torch.cat([ins, p], dim=1)  # [B, 128, 5, 20, 20]

        # channel last
        ins = rearrange(ins, "b d ... -> b ... d")  # [B, 5, 20, 20, 128]

        # save original shape of input for layer
        ins_orig_shape = ins.shape # b, 5, 20, 20, 128

        # flatten patches into sequence
        ins = rearrange(ins, "b ... d -> b (...) d")  # [B, 5*20*20, 128] -> [B, 2000, 128]
        # add learable pos encoding
        # only added to image tokens
        if self.pe_fix:
            ins += self.pos_encoding

        # append language features as sequence
        num_lang_tok = 0
        if self.add_lang:
            l = self.lang_preprocess(
                lang_emb.view(bs * self.lang_max_seq_len, self.lang_emb_dim)
            )
            l = l.view(bs, self.lang_max_seq_len, -1)
            num_lang_tok = l.shape[1]
            ins = torch.cat((l, ins), dim=1)  # [B, 77 + 2000, 128]

        # add learable pos encoding
        if not self.pe_fix:
            ins = ins + self.pos_encoding

        x = self.fc_bef_attn(ins)
        if self.self_cross_ver == 0:
            # self-attention layers
            for self_attn, self_ff in self.layers:
                x = self_attn(x) + x
                x = self_ff(x) + x

        elif self.self_cross_ver == 1:
            lx, imgx = x[:, :num_lang_tok], x[:, num_lang_tok:]

            # within image self attention
            imgx = imgx.reshape(bs * num_img, num_pat_img * num_pat_img, -1)
            for self_attn, self_ff in self.layers[: len(self.layers) // 2]:
                imgx = self_attn(imgx) + imgx
                imgx = self_ff(imgx) + imgx

            imgx = imgx.view(bs, num_img * num_pat_img * num_pat_img, -1)
            x = torch.cat((lx, imgx), dim=1)
            # cross attention
            for self_attn, self_ff in self.layers[len(self.layers) // 2 :]:
                x = self_attn(x) + x
                x = self_ff(x) + x

        else:
            assert False

        # append language features as sequence
        if self.add_lang:
            # throwing away the language embeddings
            x = x[:, num_lang_tok:]
        x = self.fc_aft_attn(x)

        # reshape back to orginal size
        x = x.view(bs, *ins_orig_shape[1:-1], x.shape[-1])  # [B, num_img, np, np, 128]
        x = rearrange(x, "b ... d -> b d ...")  # [B, 128, num_img, np, np]

        feat = []
        _feat = torch.max(torch.max(x, dim=-1)[0], dim=-1)[0] # b, 128, 5
        _feat = _feat.view(bs, -1) # b, 640
        feat.append(_feat)

        x = (
            x.transpose(1, 2) # b, 5, 128, 20, 20
            .clone()
            .view(
                bs * self.num_img, self.input_dim_before_seq, num_pat_img, num_pat_img
            )
        )
        if self.cvx_up:
            trans = self.up0(x)
            trans = trans.view(bs, self.num_img, h, w)
        else:
            u0 = self.up0(x) # b, 5, 64, 220, 220
            if self.inp_pre_con:
                u0 = torch.cat([u0, d0], dim=1) # b, 5, 128, 220, 220
            u = self.final(u0) # b, 5, 64, 220, 220

            # translation decoder
            # trans = self.trans_decoder(u).view(bs, self.num_img, h, w) # b, 5, 220, 220
            trans = self.trans_decoder(u).view(bs, self.num_img, self.trans_dim, h, w) # b, 5, 4, 220, 220

        # Below code is for feature extraction, not used for translation extraction
        if not self.no_feat:
            if self.feat_ver == 0:
                # Use only the first point from each camera (index 0)
                trans_first_point = trans[:, :, 0, :, :]  # b, 5, 220, 220 (select first point)
                hm = F.softmax(trans_first_point.detach().view(bs, self.num_img, h * w), 2).view(
                    bs * self.num_img, 1, h, w
                ) # b * 5, 1, 220, 220

                if self.cvx_up:
                    # since we donot predict u, we need to get u from xout
                    _hm = F.unfold(
                        hm,
                        kernel_size=self.img_patch_size,
                        padding=0,
                        stride=self.img_patch_size,
                    )
                    assert _hm.shape == (
                        bs * self.num_img,
                        self.img_patch_size * self.img_patch_size,
                        num_pat_img * num_pat_img,
                    )
                    _hm = torch.mean(_hm, 1)
                    _hm = _hm.view(bs * self.num_img, 1, num_pat_img, num_pat_img)
                    _u = x
                else:
                    # (bs * num_img, self.input_dim_before_seq, h, w)
                    # we use the u directly
                    _hm = hm # b * 5, 1, 220, 220
                    _u = u # b * 5, 64, 220, 220

                _feat = torch.sum(_hm * _u, dim=[2, 3]) # b * 5, 64
                _feat = _feat.view(bs, -1) # b, 320

            elif self.feat_ver == 1:
                # Use only the first point from each camera for feature extraction
                trans_first_point = trans[:, :, 0, :, :]  # b, 5, 220, 220 (select first point)
                
                # get wpt_local while testing
                if not self.training:
                    wpt_local = self.get_wpt(
                        out={"trans": trans_first_point.clone().detach()},
                        dyn_cam_info=None,
                    )

                # projection
                # (bs, 1, num_img, 2)
                wpt_img = self.get_pt_loc_on_img(
                    wpt_local.unsqueeze(1),
                    dyn_cam_info=None,
                )
                wpt_img = wpt_img.reshape(bs * self.num_img, 2)

                # add noise to wpt image while training
                if self.training:
                    wpt_img = mvt_utils.add_uni_noi(
                        wpt_img, self.wpt_img_aug * self.img_size
                    )
                    wpt_img = torch.clamp(wpt_img, 0, self.img_size - 1)

                if self.cvx_up:
                    _wpt_img = wpt_img / self.img_patch_size
                    _u = x
                    assert (
                        0 <= _wpt_img.min() and _wpt_img.max() <= x.shape[-1]
                    ), print(_wpt_img, x.shape)
                else:
                    _u = u
                    _wpt_img = wpt_img

                _wpt_img = _wpt_img.unsqueeze(1)
                _feat = select_feat_from_hm(_wpt_img, _u)[0]
                _feat = _feat.view(bs, -1)

            else:
                assert False, NotImplementedError

            feat.append(_feat) # [[b,640],[b,320]]

            feat = torch.cat(feat, dim=-1) # b, 960

            if self.rot_ver == 0:
                feat = self.feat_fc(feat) # b, 220
                out = {"feat": feat}
            elif self.rot_ver == 1:
                # features except rotation
                feat_ex_rot = self.feat_fc_ex_rot(feat) # b, 4

                # batch normalized features for rotation
                feat_rot = self.feat_fc_init_bn(feat) # b, 960
                feat_x = self.feat_fc_x(feat_rot) # b, 72

                if self.training:
                    rot_x = rot_x_y[..., 0].view(bs, 1) # b, 960
                else:
                    # sample with argmax
                    rot_x = feat_x.argmax(dim=1, keepdim=True) # b, 960

                rot_x_pe = self.feat_fc_pe(rot_x) # b, 960
                feat_y = self.feat_fc_y(feat_rot + rot_x_pe) # b, 72

                if self.training:
                    rot_y = rot_x_y[..., 1].view(bs, 1) # b, 960
                else:
                    rot_y = feat_y.argmax(dim=1, keepdim=True) # b, 960
                rot_y_pe = self.feat_fc_pe(rot_y) # b, 960
                feat_z = self.feat_fc_z(feat_rot + rot_x_pe + rot_y_pe) # b, 72
                out = {
                    "feat_ex_rot": feat_ex_rot, # b, 4w
                    "feat_x": feat_x, # b, 72
                    "feat_y": feat_y, # b, 72
                    "feat_z": feat_z, # b, 72
                }
        else:
            out = {}

        out.update({"trans": trans})

        return out

    def get_wpt(self, out, dyn_cam_info, y_q=None):
        """
        Estimate the q-values given output from mvt
        :param out: output from mvt
        """
        nc = self.num_img
        h = w = self.img_size
        bs = out["trans"].shape[0]

        # Handle the new trans_dim structure - extract waypoints for all 4 points
        # out["trans"] can have shape (bs, nc, trans_dim, h, w) where trans_dim = 4
        # or (bs, nc, h, w) if already processed
        if len(out["trans"].shape) == 5:
            # Full trans tensor with trans_dim dimension - extract all 4 points
            trans_full = out["trans"]  # (bs, nc, trans_dim, h, w)
        else:
            # Already processed to single point - expand to match expected shape
            trans_full = out["trans"].unsqueeze(2)  # (bs, nc, 1, h, w)

        # Extract waypoints for all trans_dim points
        all_pred_wpt = []
        for point_idx in range(trans_full.shape[2]):  # Loop through all 4 points
            trans_point = trans_full[:, :, point_idx, :, :]  # (bs, nc, h, w)
            
            q_trans = trans_point.view(bs, nc, h * w) # b, 5, 220 * 220
            hm = torch.nn.functional.softmax(q_trans, 2) # b, 5, 220 * 220
            hm = hm.view(bs, nc, h, w) # b, 5, 220, 220

            if dyn_cam_info is None:
                dyn_cam_info_itr = (None,) * bs
            else:
                dyn_cam_info_itr = dyn_cam_info

            pred_wpt_point = [
                self.renderer.get_max_3d_frm_hm_cube(
                    hm[i : i + 1],
                    fix_cam=True,
                    dyn_cam_info=dyn_cam_info_itr[i : i + 1]
                    if not (dyn_cam_info_itr[i] is None)
                    else None,
                )
                for i in range(bs)
            ]
            pred_wpt_point = torch.cat(pred_wpt_point, 0)
            if self.use_point_renderer:
                pred_wpt_point = pred_wpt_point.squeeze(1)
            
            all_pred_wpt.append(pred_wpt_point)

        # Stack all waypoints to get shape (bs, trans_dim, 3)
        pred_wpt = torch.stack(all_pred_wpt, dim=1)  # (bs, 4, 3)

        # Print shapes to verify we're learning 4 points
        # print(f"out['trans'] shape: {out['trans'].shape}")
        # print(f"pred_wpt shape: {pred_wpt.shape}")

        assert y_q is None

        return pred_wpt

    def free_mem(self):
        """
        Could be used for freeing up the memory once a batch of testing is done
        """
        print("Freeing up some memory")
        self.renderer.free_mem()
