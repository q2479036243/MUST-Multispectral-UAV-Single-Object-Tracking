import math
import numpy as np
from lib.models.untrack import build_untrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class UNTrack(BaseTracker):
    def __init__(self, params):
        super(UNTrack, self).__init__(params)
        network = build_untrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        
        self.num_searches = self.cfg.DATA.SEARCH.LENGTH

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            # self.z_dict1 = template
            self.memory_frames = [template.tensors]

        self.memory_masks = []
        if self.cfg.MODEL.BACKBONE.CE_LOC:  # use CE module
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.memory_masks.append(generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox))
        
        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, images, info: dict = None):
        image = images[-1] 
        H, W, _ = image.shape
        self.frame_id += 1
        searches = []
        for ind in range(self.num_searches):
            x_patch_arr, resize_factor, x_amask_arr = sample_target(images[ind], self.state, self.params.search_factor,
                                                                       output_sz=self.params.search_size)  # (x1, y1, w, h)
            search = self.preprocessor.process(x_patch_arr, x_amask_arr)
            searches.append(search.tensors)

        # --------- select memory frames ---------
        box_mask_z = None
        if self.frame_id <= self.cfg.TEST.TEMPLATE_NUMBER:
            template_list = self.memory_frames.copy()
            if self.cfg.MODEL.BACKBONE.CE_LOC:  # use CE module
                # box_mask_z = torch.cat(self.memory_masks, dim=1)
                box_mask_z = None
        else:
            template_list, box_mask_z = self.select_memory_frames()
        # --------- select memory frames ---------

        with torch.no_grad():
            out_dict = self.network.forward(template=template_list, search=searches, ce_template_mask=box_mask_z)

        if isinstance(out_dict, list):
            out_dict = out_dict[-1]
            
        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        

        # --------- save memory frames and masks ---------
        z_patch_arr, z_resize_factor, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                    output_sz=self.params.template_size)
        cur_frame = self.preprocessor.process(z_patch_arr, z_amask_arr)
        frame = cur_frame.tensors
        # mask = cur_frame.mask
        if self.frame_id > self.cfg.TEST.MEMORY_THRESHOLD:
            frame = frame.detach().cpu()
            # mask = mask.detach().cpu()
        self.memory_frames.append(frame)
        if self.cfg.MODEL.BACKBONE.CE_LOC:  # use CE module
            template_bbox = self.transform_bbox_to_crop(self.state, z_resize_factor, frame.device).squeeze(1)
            self.memory_masks.append(generate_mask_cond(self.cfg, 1, frame.device, template_bbox))
        if 'pred_iou' in out_dict.keys():      # use IoU Head
            pred_iou = out_dict['pred_iou'].squeeze(-1)
            self.memory_ious.append(pred_iou)
        # --------- save memory frames and masks ---------
        
        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                download = True
                if download:
                    seq_name = 'SC-Person-003'
                    save_dir = '/data/users/qinhaolin01/MUST-BIT/UNTrack/output/test/tracking_results/untrack/vis'
                    save_path = os.path.join(save_dir, seq_name)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    
                self.visdom.register(torch.from_numpy(x_patch_arr[:,:,[4,2,1]]).permute(2, 0, 1), 'image', 1, 'search_region')
                if download:
                    save_name = os.path.join(save_path, "%04d_search.jpg" % self.frame_id)
                    cv2.imwrite(save_name, x_patch_arr[:,:,[4,2,1]])
                
                # self.visdom.register(torch.from_numpy(self.z_patch_arr[:,:,[4,2,1]]).permute(2, 0, 1), 'image', 1, 'template')
                # if download:
                #     save_name = os.path.join(save_path, "%04d_template.jpg" % self.frame_id)
                #     cv2.imwrite(save_name, self.z_patch_arr[:,:,[4,2,1]])
                
                # self.visdom.register((image[:,:,[4,2,1]], info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')
                # if download:
                #     save_name = os.path.join(save_path, "%04d_tracking.jpg" % self.frame_id)
                #     #
                #     rgb_image = cv2.cvtColor(image[:,:,[4,2,1]], cv2.COLOR_RGB2BGR)
                #     #
                #     box_list = [int(item) for item in info['gt_bbox']]
                #     img_track = cv2.rectangle(rgb_image, (int(box_list[0]),int(box_list[1])), (int(box_list[0]+box_list[2]),int(box_list[1]+box_list[3])), color=(0,255,0), thickness=1)
                #     #
                #     box_list = [int(item) for item in self.state]
                #     img_track = cv2.rectangle(img_track, (int(box_list[0]),int(box_list[1])), (int(box_list[0]+box_list[2]),int(box_list[1]+box_list[3])), color=(0,0,255), thickness=1)
                #     # untrack_txt = os.path.join('/data/users/qinhaolin01/MUST-BIT/UNTrack/output/test/tracking_results/untrack/baseline_050/1106', seq_name+'.txt')
                #     # with open(untrack_txt, 'r') as file:
                #     #     for _ in range(self.frame_id - 1):
                #     #         next(file)  # 跳过前N-1行
                #     #         # 读取第N行
                #     #     line = next(file)
                #     # box_list = [int(item) for item in line.strip().split('\t')]
                #     # img_track = cv2.rectangle(img_track, (int(box_list[0]),int(box_list[1])), (int(box_list[0]+box_list[2]),int(box_list[1]+box_list[3])), color=(255,0,0), thickness=1)
                #     #
                #     odtrack_txt = os.path.join('/data/users/qinhaolin01/MUST-BIT/UNTrack/output/test/tracking_results/untrack/baseline_050/odtrack*', seq_name+'.txt')
                #     with open(odtrack_txt, 'r') as file:
                #         for _ in range(self.frame_id - 1):
                #             next(file)  # 跳过前N-1行
                #             # 读取第N行
                #         line = next(file)
                #     box_list = [int(item) for item in line.strip().split('\t')]
                #     img_track = cv2.rectangle(img_track, (int(box_list[0]),int(box_list[1])), (int(box_list[0]+box_list[2]),int(box_list[1]+box_list[3])), color=(255,0,0), thickness=1)
                #     #
                #     ostrack_txt = os.path.join('/data/users/qinhaolin01/MUST-BIT/UNTrack/output/test/tracking_results/untrack/baseline_050/ostrack384*', seq_name+'.txt')
                #     with open(ostrack_txt, 'r') as file:
                #         for _ in range(self.frame_id - 1):
                #             next(file)  # 跳过前N-1行
                #             # 读取第N行
                #         line = next(file)
                #     box_list = [int(item) for item in line.strip().split('\t')]
                #     img_track = cv2.rectangle(img_track, (int(box_list[0]),int(box_list[1])), (int(box_list[0]+box_list[2]),int(box_list[1]+box_list[3])), color=(0,255,255), thickness=1)
                #     #
                #     zoomtrack_txt = os.path.join('/data/users/qinhaolin01/MUST-BIT/UNTrack/output/test/tracking_results/untrack/baseline_050/zoomtrack*', seq_name+'.txt')
                #     with open(zoomtrack_txt, 'r') as file:
                #         for _ in range(self.frame_id - 1):
                #             next(file)  # 跳过前N-1行
                #             # 读取第N行
                #         line = next(file)
                #     box_list = [int(item) for item in line.strip().split('\t')]
                #     img_track = cv2.rectangle(img_track, (int(box_list[0]),int(box_list[1])), (int(box_list[0]+box_list[2]),int(box_list[1]+box_list[3])), color=(255,0,255), thickness=1)
                #     #
                #     cv2.imwrite(save_name, img_track)
                
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                # self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')
                if download:
                    save_name = os.path.join(save_path, "%04d_heatmap.jpg" % self.frame_id)
                    hotmap = pred_score_map.view(self.feat_sz, self.feat_sz).cpu().numpy()
                    heatmap = (hotmap - hotmap.min()) / (hotmap.max() - hotmap.min() + 1e-8) * 255
                    heatmap = cv2.resize(heatmap, (x_patch_arr.shape[0], x_patch_arr.shape[1]), interpolation=cv2.INTER_LINEAR)
                    heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
                    # heatmap = cv2.resize(heatmap, (x_patch_arr.shape[0], x_patch_arr.shape[1]), interpolation=cv2.INTER_LINEAR)
                    result = heatmap * 0.4 + x_patch_arr[:,:,[4,2,1]] * 0.6
                    cv2.imwrite(save_name, result)

                # if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                #     removed_indexes_s = out_dict['removed_indexes_s']
                #     removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                #     masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                #     self.visdom.register(torch.from_numpy(masked_search[:,:,[4,2,1]]).permute(2, 0, 1), 'image', 1, 'masked_search')
                #     if download:
                #         save_name = os.path.join(save_path, "%04d_masked.jpg" % self.frame_id)
                #         cv2.imwrite(save_name, masked_search[:,:,[4,2,1]])

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def select_memory_frames(self):
        num_segments = self.cfg.TEST.TEMPLATE_NUMBER
        cur_frame_idx = self.frame_id
        if num_segments != 1:
            assert cur_frame_idx > num_segments
            dur = cur_frame_idx // num_segments
            indexes = np.concatenate([
                np.array([0]),
                np.array(list(range(num_segments))) * dur + dur // 2
            ])
        else:
            indexes = np.array([0])
        indexes = np.unique(indexes)

        select_frames, select_masks = [], []
        
        for idx in indexes:
            frames = self.memory_frames[idx]
            if not frames.is_cuda:
                frames = frames.cuda()
            select_frames.append(frames)
            
            if self.cfg.MODEL.BACKBONE.CE_LOC:
                box_mask_z = self.memory_masks[idx]
                # select_masks.append(box_mask_z.cuda())
        
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            # return select_frames, torch.cat(select_masks, dim=1)
            return select_frames, None
        else:
            return select_frames, None
    
    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights

def get_tracker_class():
    return UNTrack
