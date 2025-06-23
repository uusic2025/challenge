import os

# cls

source_path = "data/classification"

for dataset in os.listdir(source_path):
    dataset_path = os.path.join(source_path, dataset)
    if not os.path.isdir(dataset_path):
        continue
    print(f"Processing dataset: {dataset}")

    train_txt_path = os.path.join(dataset_path, "train.txt")
    val_txt_path = os.path.join(dataset_path, "val.txt")
    test_txt_path = os.path.join(dataset_path, "test.txt")

    with open(train_txt_path, "w") as train_file, open(val_txt_path, "w") as val_file, open(
        test_txt_path, "w"
    ) as test_file:

        all_images_list = []
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if not file.endswith(".yaml") and not file.endswith(".txt"):
                    file_path = os.path.join(root.split("/")[-1], file)
                    all_images_list.append(file_path)
        num_images = len(all_images_list)
        train_num = int(num_images * 0.7)
        val_num = int(num_images * 0.2)
        test_num = num_images - train_num - val_num
        import random
        random.seed(42)
        random.shuffle(all_images_list)
        train_images = all_images_list[:train_num]
        val_images = all_images_list[train_num : train_num + val_num]
        test_images = all_images_list[train_num + val_num :]
        for image_path in train_images:
            train_file.write(f"{image_path}\n")
        for image_path in val_images:
            val_file.write(f"{image_path}\n")
        for image_path in test_images:
            test_file.write(f"{image_path}\n")
    print(f"Train, Val, Test txt files created for {dataset} dataset.")
    print(f"Train images: {len(train_images)}, Val images: {len(val_images)}, Test images: {len(test_images)}")
    print("-" * 50)
# seg

source_path = "data/segmentation"

for dataset in os.listdir(source_path):
    dataset_path = os.path.join(source_path, dataset)
    if not os.path.isdir(dataset_path):
        continue
    print(f"Processing dataset: {dataset}")

    train_txt_path = os.path.join(dataset_path, "train.txt")
    val_txt_path = os.path.join(dataset_path, "val.txt")
    test_txt_path = os.path.join(dataset_path, "test.txt")

    with open(train_txt_path, "w") as train_file, open(val_txt_path, "w") as val_file, open(
        test_txt_path, "w"
    ) as test_file:

        all_images_list = []
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if not file.endswith(".yaml") and not file.endswith(".txt"):
                    if "mask" in root:
                        continue
                    all_images_list.append(file)
        num_images = len(all_images_list)
        train_num = int(num_images * 0.7)
        val_num = int(num_images * 0.2)
        test_num = num_images - train_num - val_num
        import random
        random.seed(42)
        random.shuffle(all_images_list)
        train_images = all_images_list[:train_num]
        val_images = all_images_list[train_num : train_num + val_num]
        test_images = all_images_list[train_num + val_num :]
        for image_path in train_images:
            train_file.write(f"{image_path}\n")
        for image_path in val_images:
            val_file.write(f"{image_path}\n")
        for image_path in test_images:
            test_file.write(f"{image_path}\n")
    print(f"Train, Val, Test txt files created for {dataset} dataset.")
    print(f"Train images: {len(train_images)}, Val images: {len(val_images)}, Test images: {len(test_images)}")
    print("-" * 50)