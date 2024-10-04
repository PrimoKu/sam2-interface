import torch
from sam2.build_sam import build_sam2_video_predictor

class SAM2Predictor:
    def __init__(self):
        self.predictor = None
        self.inference_state = None

    def initialize_predictor(self, video_dir, progress_callback=None):
        sam2_checkpoint = "../external/sam2/checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        
        if progress_callback:
            progress_callback("Selecting computation device...")

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"using device: {device}")

        if device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif device.type == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )
        
        if progress_callback:
            progress_callback("Building SAM2 predictor...")

        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
        
        if progress_callback:
            progress_callback("Initializing inference state...")

        self.inference_state = self.predictor.init_state(video_path=video_dir)
        self.predictor.reset_state(self.inference_state)
        
        if progress_callback:
            progress_callback("Initialization complete.")

    def propagate_masks(self, start_frame_idx=0, max_frame_num_to_track=None, progress_callback=None):
        video_segments = {}
        for i, (out_frame_idx, out_obj_ids, out_mask_logits) in enumerate(self.predictor.propagate_in_video(self.inference_state, start_frame_idx=start_frame_idx, max_frame_num_to_track=max_frame_num_to_track)):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            if progress_callback:
                progress_callback(out_frame_idx)
        return video_segments

    def generate_mask_with_points(self, frame_idx, obj_id, coords, labels):
        _, _, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=coords,
            labels=labels,
        )
        return out_mask_logits
    
    def generate_mask_with_box(self, frame_idx, obj_id, box):
        _, _, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=box
        )
        return (out_mask_logits[obj_id] > 0.0).cpu().numpy()
    
    def reset_state(self):
        self.predictor.reset_state(self.inference_state)
