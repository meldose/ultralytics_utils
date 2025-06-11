import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

from ultralytics.utils import LOGGER, TQDM, yaml_load, yaml_save
from ultralytics.data.converter import merge_multi_segment


def setup_dir(output_dir: Path, overwrite: bool) -> tuple[Path, int]:
    # increment if save directory already exists
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
        LOGGER.info(f"Overwrite an existing dataset '{output_dir}'")

    (output_dir / "labels").mkdir(parents=True, exist_ok=True)

    num_files = sum(1 for _ in (output_dir / "labels").rglob("*.txt"))
    LOGGER.info(f"Found {num_files} existing images")

    return output_dir, num_files


def mask_to_polygons(mask: np.ndarray) -> list:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Convert contours to polygon coordinates
    polygons = []

    for contour in contours:
        # Flatten the contour and convert to list
        polygon = contour.flatten().tolist()
        # Only add if we have enough points
        if len(polygon) >= 6:  # At least 3 points (x,y pairs)
            polygons.append(polygon)

    return polygons


def rle_to_polygons(
    segm: Dict[str, Any], height: int, width: int
) -> List[List[int]]:
    """Convert RLE to polygon coordinates."""
    from pycocotools import mask as mask_utils

    # Decode RLE to binary mask
    if "counts" in segm and isinstance(segm["counts"], list):
        # Convert list counts to RLE string format
        rle = mask_utils.frPyObjects(segm, height, width)
    else:
        rle = segm

    binary_mask = mask_utils.decode(rle)

    return mask_to_polygons(binary_mask)


def get_class_names(
    input_dir: Path, coco_annotations_file: Union[Path, None] = None
) -> List[str]:
    """
    Get class names from COCO dataset annotations.

    Args:
        input_dir (Path, optional): Path to directory containing COCO dataset
                                    annotation files.
        coco_annotations_file (Path, optional): Path to COCO annotation file.

    Output:
        List of class names.
    """

    class_names = []

    if coco_annotations_file is None:
        json_files = sorted(input_dir.glob("*.json"))
    else:
        json_files = [input_dir / coco_annotations_file]

    for json_file in json_files:
        LOGGER.info(f"read '{json_file}'")

        with open(json_file) as f:
            data = json.load(f)

        class_names = [x["name"] for x in data["categories"]]

    return class_names


def create_segment(segm: Dict[str, Any], height: int, width: int) -> List[int]:
    if isinstance(segm, dict):
        segm = rle_to_polygons(segm, height, width)

    if len(segm) == 0:
        return []

    if len(segm) > 1:
        s = merge_multi_segment(segm)
        s = np.concatenate(s, axis=0)
    else:
        s = [j for i in segm for j in i]
        s = np.array(s).reshape(-1, 2)

    size = np.array([width, height])
    s = (s / size).reshape(-1).tolist()

    return s


def create_keypoints(
    keypoints: List[int], height: int, width: int
) -> List[int]:
    keypoints = np.array(keypoints).reshape(-1, 3)
    size = np.array([width, height, 1])

    return keypoints / size.reshape(-1).tolist()


def convert_coco(
    input_dir: Path,
    output_dir: Path,
    coco_annotations_file: Union[Path, None] = None,
    keep_filenames: bool = False,
    overwrite: bool = False,
    cls_mapping: Union[Dict[str, int], None] = None,
    use_segments: bool = False,
    use_keypoints: bool = False,
):
    """
    Converts COCO dataset annotations to a YOLO annotation format suitable for
    training YOLO models.

    Args:
        input_dir (Path): Path to directory containing COCO dataset
                          annotation files.
        output_dir (Path): Path to directory to save results to.
        coco_annotations_file (Path, optional): Path to COCO annotation file.
        keep_filenames (bool, optional): Whether to keep the original filenames.
        overwrite (bool, optional): Whether to overwrite existing dataset.
        cls_mapping (dict, optional): Mapping of class names to class IDs.
        use_segments (bool, optional): Whether to include segmentation masks
                                       in the output.
        use_keypoints (bool, optional): Whether to include keypoint
                                        annotations in the output.

    Output:
        Generates output files in the specified output directory.
    """
    LOGGER.info(
        f"Convert '{input_dir}' to '{output_dir}'\n"
        f"\tcoco_annotations_file={coco_annotations_file}\n"
        f"\tkeep_filenames={keep_filenames}\n"
        f"\tuse_segments={use_segments}\n"
        f"\tuse_keypoints={use_keypoints}\n"
        f"\tcls_mapping={cls_mapping}\n"
    )

    output_dir, num_files = setup_dir(output_dir, overwrite=overwrite)

    if coco_annotations_file is None:
        json_files = sorted(input_dir.glob("*.json"))
    else:
        json_files = [input_dir / coco_annotations_file]

    LOGGER.info(f"Found {len(json_files)} new json files with annotations")

    for json_file in json_files:
        LOGGER.info(f"read '{json_file}'")

        with open(json_file) as f:
            data = json.load(f)

        cls_id_to_name = {x["id"]: x["name"] for x in data["categories"]}
        LOGGER.info(cls_id_to_name)

        # Create image dict
        images = {f"{x['id']:d}": x for x in data["images"]}
        # Create image-annotations dict
        img_to_anns = defaultdict(list)

        for ann in data["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)

        LOGGER.info(f"add {len(img_to_anns)} new images")

        # no, we can't -- ultrlytics is too stupid to follow links....
        # # if input and output dir are the same and we keep the file names,
        # # the images folder can just symlink to rgb folder
        # if input_dir == output_dir and keep_filenames:
        #     (input_dir / "images").symlink_to(output_dir / "rgb")

        # Write labels file
        for img_id, anns in TQDM(
            img_to_anns.items(), desc=f"annotations {json_file}"
        ):
            img = images[f"{img_id:d}"]
            height, width = img["height"], img["width"]

            filename = Path(img["file_name"]).name
            src_img_path = input_dir / "rgb" / filename

            if keep_filenames:
                save_filename = filename
            else:
                save_filename = f"{num_files:06d}.jpg"

            num_files += 1

            dst_img_path = output_dir / "images" / save_filename

            print(img_id, len(anns), filename, src_img_path, dst_img_path)

            # if src_img_path != dst_img_path and not (
            #     input_dir == output_dir and keep_filenames
            # ):
            if src_img_path != dst_img_path:
                dst_img_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src_img_path, dst_img_path)

            bboxes = list()
            segments = list()
            keypoints = list()

            for ann in anns:
                # The COCO box format is
                # [top left x, top left y, width, height]
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= width  # normalize x
                box[[1, 3]] /= height  # normalize y

                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                cls_id = ann["category_id"]
                cls_name = cls_id_to_name[cls_id]

                if cls_mapping is not None:
                    cls = cls_mapping[cls_name]
                else:
                    cls = cls_id - 1

                box = [cls] + box.tolist()

                if box not in bboxes:
                    bboxes.append(box)

                    if use_segments and ann.get("segmentation") is not None:
                        s = create_segment(
                            ann["segmentation"], height=height, width=width
                        )
                        segments.append(([cls] + s) if len(s) > 0 else [])

                    if use_keypoints and ann.get("keypoints") is not None:
                        k = create_keypoints(
                            ann["keypoints"], height=height, width=width
                        )
                        keypoints.append(box + k)

            # Write
            labels_path = output_dir / "labels" / save_filename

            with open(labels_path.with_suffix(".txt"), "w") as file:
                for i in range(len(bboxes)):
                    if use_keypoints:
                        line = (*(keypoints[i]),)  # cls, box, keypoints
                    else:
                        line = (
                            *(
                                segments[i]
                                if use_segments and len(segments[i]) > 0
                                else bboxes[i]
                            ),
                        )

                    file.write(("%g " * len(line)).rstrip() % line + "\n")

    LOGGER.info(f"Save dataset to '{output_dir.resolve()}'\n")


def main(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    coco_annotations_file: Union[Path, None] = None,
    overwrite: bool = False,
    keep_filenames: bool = False,
):
    if len(input_dir) > 1 and keep_filenames is True:
        raise ValueError(
            "Keep filenames must be False if merging multiple directories"
        )

    class_names = set()

    for input_dir in args.input_dir:
        for split in ("train", "val", "test"):
            split_class_names = get_class_names(
                input_dir=input_dir / split,
                coco_annotations_file=coco_annotations_file,
            )
            class_names.update(split_class_names)

    class_names = sorted(class_names)
    class_name_to_id = {
        class_name: idx for idx, class_name in enumerate(class_names)
    }

    LOGGER.info("Class mapping:")

    for class_name, class_id in class_name_to_id.items():
        LOGGER.info(f"{class_name:<30} -> {class_id}")

    for input_dir in args.input_dir:
        LOGGER.info(f"Convert {input_dir}")

        if output_dir is None:
            output_dir = input_dir

        if overwrite:
            if input_dir == output_dir:
                raise ValueError(
                    "Overwrite mode is not allowed if input_dir and output_dir "
                    "are the same"
                )

            if output_dir.exists():
                shutil.rmtree(output_dir)

        for split in ("train", "val", "test"):
            LOGGER.info(f"Convert '{split}' split")
            convert_coco(
                input_dir=input_dir / split,
                output_dir=output_dir / split,
                coco_annotations_file=coco_annotations_file,
                keep_filenames=keep_filenames,
                overwrite=overwrite,
                cls_mapping=class_name_to_id,
                use_segments=True,
                use_keypoints=False,
            )

        overwrite = False

        names = {id: name for name, id in class_name_to_id.items()}

    # dataset_name = input_dir.parent.name
    # yaml_dir = output_dir.parent
    dataset_name = output_dir.parent.name
    yaml_dir = output_dir.parent

    # Create config for training and validation
    yaml_path = yaml_dir / f"{dataset_name}.yaml"

    if yaml_path.exists():
        yaml_data = yaml_load(yaml_path)
    else:
        yaml_data = {"path": str(output_dir), "names": names}

    yaml_data["train"] = "train/images"
    yaml_data["val"] = "val/images"
    yaml_data["test"] = "test/images"

    LOGGER.info(f"Save YOLO config to '{yaml_path}'")
    yaml_save(yaml_path, yaml_data)

    # Create config for evaluation on the test split
    yaml_path = yaml_dir / f"{dataset_name}_test.yaml"

    yaml_data["val"] = "test/images"

    LOGGER.info(f"Save YOLO config for test split to '{yaml_path}'")
    yaml_save(yaml_path, yaml_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        nargs="+",
        help="path(s) to original COCO dataset",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        help="path to converted dataset in ultralytics format",
    )
    parser.add_argument(
        "--coco-annotations-file",
        type=Path,
        default=None,
        help="path to COCO annotations file",
    )
    parser.add_argument(
        "--overwrite",
        type=bool,
        default=False,
        help="overwrite existing dataset",
    )
    parser.add_argument(
        "--keep-filenames", action="store_true", help="keep original filenames"
    )

    args = parser.parse_args()

    main(**vars(args))
