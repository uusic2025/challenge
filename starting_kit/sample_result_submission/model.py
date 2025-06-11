# This is a simplified sample code submission for a combined segmentation and classification task.
# It simulates predictions using basic logic and saves output in the required format.

import os
import json
import time
import random
import numpy as np
from PIL import Image


class Model:
    def __init__(self):
        """Initialize the model (can be replaced with real model loading)."""
        self.network = None
        pass

    def predict_segmentation_and_classification(self, data_list, input_dir, output_dir):
        """
        Perform dummy segmentation and classification on given images.
        
        Args:
            data_list (list): List of dicts, each containing task type and image path.
            input_dir (str): Root path to input images.
            output_dir (str): Root path where outputs (masks or predictions) should be saved.
        
        Returns:
            dict: Classification predictions (for classification task only).
        """
        class_predictions = {}

        for data_dict in data_list:
            task = data_dict['task']
            dataset_name = data_dict['dataset_name']
            organ = data_dict['organ']

            image_path = os.path.join(input_dir, data_dict['img_path_relative'])
            img = Image.open(image_path).convert('L')  # Load as grayscale

            if task == 'classification':
                if dataset_name=='Breast_luminal':
                    num_classes = 4
                else:
                    num_classes = 2
                # probability = self.network(img)

                # Simulate classification using a random threshold
                probability = random.uniform(0, 1)
                prediction = int(probability > 0.5)
                if num_classes==2:
                    probability = [1 - probability, probability]
                elif num_classes==3:
                    probability = [1 - probability, probability, 0]
                elif num_classes==4:
                    probability = [1 - probability, probability, 0, 0]

                class_predictions[data_dict['img_path_relative']] = {
                    'probability': probability,
                    'prediction': prediction
                }

            elif task == 'segmentation':
                #binary_mask = self.network(img)

                # Simulate segmentation by thresholding grayscale image at its mean
                img_array = np.array(img)
                threshold = img_array.mean()
                binary_mask = (img_array > threshold).astype(np.uint8) * 255
                mask_img = Image.fromarray(binary_mask)

                # Save the generated mask
                save_path = os.path.join(output_dir, data_dict['img_path_relative'].replace('img', 'mask'))
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                mask_img.save(save_path)

        return class_predictions


def main():
    """Main function simulating Codabench ingestion program behavior."""
    # Define paths
    input_dir = '/path/to/Challenge_Data_Private_v2_fully_anonymized/Val'
    data_list_path = '/path/to/dataset_json_fingerprints_v3/private_val_for_participants.json'
    output_dir = 'sample_result_submission'
    
    print('-' * 10)
    print('Starting inference program.')

    start_time = time.time()

    # Load input metadata
    with open(os.path.join(data_list_path), 'r') as f:
        data_list = json.load(f)

    # Initialize model
    print('Initializing model...')
    model = Model()

    # Generate predictions
    print('Generating predictions...')
    predictions = model.predict_segmentation_and_classification(data_list, input_dir, output_dir)

    # Save classification predictions
    class_output_path = os.path.join(output_dir, 'classification.json')
    with open(class_output_path, 'w') as f:
        json.dump(predictions, f, indent=4)


if __name__ == '__main__':
    main()
