import argparse  # imported argparse module 
import shutil # imported shutil module 
import random # imported random  module 
from pathlib import Path # imported Path module 
import yaml # imported yaml module 


# function for spliting the dataset
def split_dataset(
    dataset_path: Path,
    train_ratio: float = 0.7,
    test_ratio: float = 0.1
):
    """Split dataset into train/valid/test sets.
    
    Args:
        dataset_path (str): Path to dataset directory
        train_ratio (float): Ratio of data for training (0-1)
        test_ratio (float): Ratio of data for testing (0-1)
        Valid ratio will be 1 - train_ratio - test_ratio
    """
    # Get all image files
    image_files = [p.name for p in (dataset_path / "images").iterdir()]
    random.shuffle(image_files)

    # Calculate split
    num_train = int(len(image_files) * train_ratio)
    num_valid = int(len(image_files) * test_ratio)
    num_test = len(image_files) - num_train - num_valid

    train_files = image_files[:num_train]
    valid_files = image_files[num_train:num_train + num_valid]
    test_files = image_files[num_train + num_valid:]

# function for copying the files
    def copy_files(files, split):
        split_dir = dataset_path / split
        if split_dir.exists():
            shutil.rmtree(split_dir)

        split_dir.mkdir(parents=True, exist_ok=True)

        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)

        existed = 0
        created = 0

        for img_file in files:
            # Copy image
            src_img_path = dataset_path / "images" / img_file
            dst_img_path = split_dir / "images" / img_file
            shutil.copy2(src_img_path, dst_img_path)

            # Copy corresponding label if exists
            src_label_path = (
                dataset_path / "labels" / img_file
            ).with_suffix(".txt")
            dst_label_path = split_dir / "labels" / src_label_path.name

            if src_label_path.exists():
                shutil.copy2(src_label_path, dst_label_path)
                existed += 1
            else:
                print(f"Label file {src_label_path} does not exist!")
                dst_label_path.touch()
                created += 1

        print(f"Copied: {existed} labels, created: {created} labels")

    # Execute copying
    for split, files in (
        ("train", train_files),
        ("val", valid_files),
        ("test", test_files)
    ):
        print(f"Copying {split} set...")
        copy_files(files, split)

    print(
        f"Split complete! Train: {len(train_files)}, "
        f"Validation: {len(valid_files)}, "
        f"Test: {len(test_files)} images"
    )

    # Create config files
    dataset_name = dataset_path.name
    print(f"Creating config files for {dataset_name}...")
    
    yaml_files = list(dataset_path.glob("*.yaml"))

    if len(yaml_files) == 0:
        raise ValueError(f"No config file found at {dataset_path.as_posix()}")
    elif len(yaml_files) > 1:
        raise ValueError(
            f"Multiple config files found at {dataset_path.as_posix()}"
        )

    original_cfg_path = yaml_files[0]

    print(f"Found config file at {original_cfg_path.as_posix()}")

    with open(original_cfg_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    print(f"Loaded config: {cfg}")

    # Create default config
    cfg["train"] = "train"

    cfg_path = dataset_path / f"{dataset_name}.yaml"

    with open(cfg_path, "w") as f:
        cfg["val"] = "val"
        yaml.dump(cfg, f)

    print(f"Saved config file at {cfg_path.as_posix()}")

    # Create config for test set
    cfg_test_path = dataset_path / f"{dataset_name}-test.yaml"

    with open(cfg_test_path, "w") as f:
        cfg["val"] = "test"
        yaml.dump(cfg, f)

    print(f"Saved config file at {cfg_test_path.as_posix()}")

    print("Done!")

# calling the main function 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split dataset into train/valid/test sets"
    )
    parser.add_argument(
        "--dataset-path",
        "--data",
        type=Path,
        required=True,
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--train-ratio",
        "--train",
        type=float,
        default=0.8,
        help="Ratio of data for training (0-1)"
    )
    parser.add_argument(
        "--test-ratio",
        "--test",
        type=float,
        default=0.1,
        help="Ratio of data for testing (0-1)"
    )
    args = parser.parse_args()

    split_dataset(
        dataset_path=args.dataset_path,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio
    )
