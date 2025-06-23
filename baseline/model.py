# sample_result_submission/model.py

import os
import cv2
import json
from PIL import Image
import numpy as np
from tqdm import tqdm


import torch
from config import get_config
from networks.omni_vision_transformer import OmniVisionTransformer as ViT_omni
from datasets.dataset import CenterCropGenerator


organ_to_position_map = {
    'Breast': 'breast',
    'Heart': 'cardiac',
    'Thyroid': 'thyroid',
    'Head': 'head',
    'Kidney': 'kidney',
    'Appendix': 'appendix',
    'Liver': 'liver',
}


position_prompt_one_hot_dict = {
    "breast":   [1, 0, 0, 0, 0, 0, 0, 0],
    "cardiac":  [0, 1, 0, 0, 0, 0, 0, 0],
    "thyroid":  [0, 0, 1, 0, 0, 0, 0, 0],
    "head":     [0, 0, 0, 1, 0, 0, 0, 0],
    "kidney":   [0, 0, 0, 0, 1, 0, 0, 0],
    "appendix": [0, 0, 0, 0, 0, 1, 0, 0],
    "liver":    [0, 0, 0, 0, 0, 0, 1, 0],
    "indis":    [0, 0, 0, 0, 0, 0, 0, 1]
}


task_prompt_one_hot_dict = {
    "segmentation": [1, 0],
    "classification": [0, 1]
}

organ_to_nature_map = {
    'Breast': 'tumor',
    'Heart': 'organ',
    'Thyroid': 'tumor',
    'Head': 'organ',
    'Kidney': 'organ',
    'Appendix': 'organ',
    'Liver': 'organ',
}

nature_prompt_one_hot_dict = {
    "tumor": [1, 0],
    "organ": [0, 1],
}

type_prompt_one_hot_dict = {
    "whole": [1, 0, 0],
    "local": [0, 1, 0],
    "location": [0, 0, 1],
}



class Model:
    def __init__(self):
        print("Initializing model...")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class Args:
            cfg = 'configs/swin_tiny_patch4_window7_224_lite.yaml' # 你的配置文件名
            img_size = 224
            prompt = True

            opts = None
            batch_size = None
            zip = False
            cache_mode = None
            resume = None
            accumulation_steps = None
            use_checkpoint = False
            amp_opt_level = ''
            tag = None
            eval = False
            throughput = False
        
        args = Args()
        self.args = args
        config = get_config(args)

        self.network = ViT_omni(config, prompt=args.prompt).to(self.device)


        snapshot_path = 'exp_out/trial_2/best_model.pth'
        pretrained_dict = torch.load(snapshot_path, map_location=self.device)
        

        if "module.swin.layers_task_cls_head_4cls.0.weight" in pretrained_dict:
            # DDP v2 (state_dict in a dict)
            if 'model' in pretrained_dict:
                pretrained_dict = pretrained_dict['model']

            new_state_dict = {}
            for k, v in pretrained_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v  # remove `module.`
                else:
                    new_state_dict[k] = v
            self.network.load_state_dict(new_state_dict)
        elif "swin.layers_task_cls_head_4cls.0.weight" in pretrained_dict:
            # DDP v1
            self.network.load_state_dict(pretrained_dict)
        else:
            # load from self (omni_test.py)
            import copy
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "module." not in k:
                    full_dict["module."+k] = v
                    del full_dict[k]
            self.network.load_state_dict(full_dict)

        self.network.eval()
        
        self.transform = CenterCropGenerator(output_size=[args.img_size, args.img_size])

        print("Model initialized.")

    def predict_segmentation_and_classification(self, data_list, input_dir, output_dir):
        class_predictions = {}

        for data_dict in tqdm(data_list, desc="Processing images"):
            img_path = os.path.join(input_dir, data_dict['img_path_relative'])
            task = data_dict['task']
            dataset_name = data_dict['dataset_name']
            organ_name = data_dict['organ']

            img = Image.open(img_path).convert('RGB')
            original_size = img.size # (width, height)
            img_np = np.array(img)
            
            sample = {'image': img_np / 255.0, 'label': np.zeros(img_np.shape[:2])}
            processed_sample = self.transform(sample)
            image_tensor = processed_sample['image'].to(self.device) # shape: [1, H, W, C]

            with torch.no_grad():
                if self.args.prompt:
                    task_p_vec = task_prompt_one_hot_dict[task]
                    task_prompt = torch.tensor(task_p_vec, dtype=torch.float).unsqueeze(0).to(self.device)

                    position_key = organ_to_position_map.get(organ_name, 'indis')
                    position_p_vec = position_prompt_one_hot_dict[position_key]
                    position_prompt = torch.tensor(position_p_vec, dtype=torch.float).unsqueeze(0).to(self.device)

                    nature_key = organ_to_nature_map.get(organ_name, 'organ')
                    nature_p_vec = nature_prompt_one_hot_dict[nature_key]
                    nature_prompt = torch.tensor(nature_p_vec, dtype=torch.float).unsqueeze(0).to(self.device)
                    
                    type_p_vec = type_prompt_one_hot_dict["whole"]
                    type_prompt = torch.tensor(type_p_vec, dtype=torch.float).unsqueeze(0).to(self.device)

                    model_input = (image_tensor, position_prompt, task_prompt, type_prompt, nature_prompt)
                    outputs_tuple = self.network(model_input)
                else:
                    outputs_tuple = self.network(image_tensor)

            if task == 'classification':
                if dataset_name == 'Breast_luminal':
                    num_classes = 4
                    logits = outputs_tuple[2]
                else:
                    num_classes = 2
                    logits = outputs_tuple[1]

                probabilities = torch.softmax(logits, dim=1).cpu().numpy().flatten()
                prediction = int(np.argmax(probabilities))
                
                class_predictions[data_dict['img_path_relative']] = {
                    'probability': probabilities.tolist(),
                    'prediction': prediction
                }

            elif task == 'segmentation':
                seg_logits = outputs_tuple[0]
                
                seg_pred = torch.argmax(torch.softmax(seg_logits, dim=1), dim=1).squeeze(0)
                binary_mask_224 = seg_pred.cpu().numpy().astype(np.uint8) * 255
                
                resized_mask = cv2.resize(
                    binary_mask_224, 
                    original_size, 
                    interpolation=cv2.INTER_NEAREST
                )
                
                mask_img = Image.fromarray(resized_mask)

                save_path = os.path.join(output_dir, data_dict['img_path_relative'].replace('img', 'mask'))
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                mask_img.save(save_path)

        with open(os.path.join(output_dir, 'classification.json'), 'w') as f:
            json.dump(class_predictions, f, indent=4)


if __name__ == '__main__':
    input_dir = 'data/Val/'
    data_list_path = 'data/Val/private_val_for_participants.json'
    output_dir = 'exp_out/sample_result_submission'
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(data_list_path, 'r') as f:
        data_list = json.load(f)

    model = Model()
    model.predict_segmentation_and_classification(data_list, input_dir, output_dir)
    print("Inference completed.")