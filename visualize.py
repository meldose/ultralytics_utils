import argparse  # imported argpasemodule
import os # imported os module
from collections import defaultdict # imported defaultdict module
from pathlib import Path # imported Path module
from PIL import Image, ImageOps # imported Image, Image Opsmodule
from typing import Dict, List, Optional # imported  Dict, List , Optional module

import numpy as np # imported numpy module

from ultralytics.data.augment import Format
from ultralytics.data.utils import (
    IMG_FORMATS,
    check_cls_dataset,
    check_det_dataset,
    clean_url,
    exif_size,
    FORMATS_HELP_MSG,
    img2label_paths,
)

from ultralytics.utils import LOGGER, emojis
from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import segments2boxes, resample_segments

from ultralytics_utils.utils.plotting import plot_images


# function ofr visualization parser
def get_visualization_parser():
    parser = argparse.ArgumentParser(description="Visualize image with labels")
    parser.add_argument(
        "--image", type=Path, help="Path to image file or directory"
    )
    parser.add_argument(
        "--label", type=Path, help="Path to label file or directory"
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Path to dataset YAML config file",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Task to visualize",
        choices=["classify", "detect", "segment", "pose", "obb"],
    )
    parser.add_argument(
        "--class-names",
        nargs="*",
        help="List of class names in order. "
        "If not provided, will use enumerated classes",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        help="List of class indices in order. "
        "If not provided, will use all classes",
    )
    parser.add_argument(
        "--masks-mode",
        type=str,
        default="masks",
        help="Masks mode, 'masks' or 'contours'",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Path to output directory",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    return parser

# function for selecting the classes 
def select_classes(
    names: Dict[int, str],
    class_names: Optional[List[str]],
    classes: Optional[List[int]],
) -> Dict[int, str]:
    """
    Select classes from names by names or indices.

    Args:
        names: Dict of class names
        class_names: List of class names to select
        classes: List of class indices to select
    """
    if class_names is not None and classes is not None:
        raise ValueError("Cannot provide both class_names and classes")

    if class_names is not None:
        LOGGER.info(f"Select classes by names: {class_names}")
        names = {i: name for i, name in names.items() if name in class_names}

    if classes is not None:
        LOGGER.info(f"Select classes by indices: {classes}")
        names = {i: name for i, name in names.items() if i in args.classes}

    return names

#  function for verifying the image 
def _verify_image(image_file: Path):
    # Verify images
    image = Image.open(image_file)
    image.verify()  # PIL verify
    shape = exif_size(image)  # image size
    shape = (shape[1], shape[0])  # hw

    if (shape[0] < 10) or (shape[1] < 10):
        raise ValueError(
            f"ERROR ❌ {image_file}: image size {shape} <10 pixels"
        )

    if image.format.lower() not in IMG_FORMATS:
        raise ValueError(
            f"ERROR ❌ {image_file}: invalid image format "
            f"{image.format}. {FORMATS_HELP_MSG}"
        )

    if image.format.lower() in {"jpg", "jpeg"}:
        with open(image_file, "rb") as f:
            f.seek(-2, 2)

            if f.read() != b"\xff\xd9":  # corrupt JPEG
                ImageOps.exif_transpose(Image.open(image_file)).save(
                    image_file, "JPEG", subsampling=0, quality=100
                )
                LOGGER.warning(
                    f"WARNING ⚠️ {image_file}: corrupt JPEG restored and saved"
                )

    return Image.open(image_file)

# function for verifying the label
def _verify_label(
    label_file: Path,
    use_segments: bool,
    use_obb: bool,
    use_keypoints: bool,
    nkpt: int,
    ndim: int,
    single_cls: bool,
):
    segments, keypoints = list(), None
    label = np.zeros(
        (0, (5 + nkpt * ndim) if use_keypoints else 5), dtype=np.float32
    )

    if os.path.isfile(label_file):
        with open(label_file, encoding="utf-8") as f:
            label = [x.split() for x in f.read().strip().splitlines() if len(x)]

            if any(len(x) > 6 for x in label) and (
                not use_keypoints
            ):  # is segment
                classes = np.array([x[0] for x in label], dtype=np.float32)
                segments = [
                    np.array(x[1:], dtype=np.float32).reshape(-1, 2)
                    for x in label
                ]  # (cls, xy1...)
                label = np.concatenate(
                    (classes.reshape(-1, 1), segments2boxes(segments)), 1
                )  # (cls, xywh)

            label = np.array(label, dtype=np.float32)

        if len(label):
            if use_keypoints:
                if label.shape[1] != (5 + nkpt * ndim):
                    raise ValueError(
                        f"labels require {(5 + nkpt * ndim)} columns each"
                    )
                points = label[:, 5:].reshape(-1, ndim)[:, :2]
            else:
                if label.shape[1] != 5:
                    raise ValueError(
                        f"labels require 5 columns, {label.shape[1]} columns "
                        "detected"
                    )
                points = label[:, 1:]

            if points.max() > 1:
                raise ValueError(
                    f"non-normalized or out of bounds coordinates "
                    f"{points[points > 1]}"
                )

            if label.min() < 0:
                raise ValueError(f"negative label values {label[label < 0]}")

            # All labels
            if single_cls:
                label[:, 0] = 0

            _, i = np.unique(label, axis=0, return_index=True)

            if len(i) < len(label):  # duplicate row check
                label = label[i]  # remove duplicates

                if segments:
                    segments = [segments[x] for x in i]

                LOGGER.warning(
                    f"WARNING ⚠️ {label_file}: {len(label) - len(i)} "
                    "duplicate labels removed"
                )

    if use_keypoints:
        keypoints = label[:, 5:].reshape(-1, nkpt, ndim)

        if ndim == 2:
            kpt_mask = np.where(
                (keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0
            ).astype(np.float32)

            keypoints = np.concatenate(
                [keypoints, kpt_mask[..., None]], axis=-1
            )  # (nl, nkpt, 3)

    cls = label[:, 0:1]  # n, 1
    bboxes = label[:, 1:]  # n, 4

    return cls, bboxes, segments, keypoints

# function for getting the labels 
def _get_labels(
    image_file: Path,
    label_file: Path,
    use_segments: bool,
    use_obb: bool,
    use_keypoints: bool,
    nkpt: int,
    ndim: int,
    single_cls: bool,
):
    """Verify one image-label pair."""
    # Verify images
    try:
        image = _verify_image(image_file)
    except ValueError as e:
        raise ValueError(f"Image {image_file} is invalid: {e}")

    # Verify label
    try:
        cls, bboxes, segments, keypoints = _verify_label(
            label_file=label_file,
            use_segments=use_segments,
            use_obb=use_obb,
            use_keypoints=use_keypoints,
            nkpt=nkpt,
            ndim=ndim,
            single_cls=single_cls,
        )
    except ValueError as e:
        raise ValueError(f"Label {label_file} is invalid: {e}")

    segments = np.stack(resample_segments(segments), axis=0)

    instances = Instances(
        bboxes=bboxes,
        segments=segments,
        keypoints=keypoints,
        bbox_format="xywh",
        normalized=True,
    )
    labels = {
        "img": np.array(image),
        "cls": cls,
        "instances": instances,
    }

    return Format(
        bbox_format="xywh",
        normalize=True,
        return_mask=use_segments,
        return_keypoint=use_keypoints,
        return_obb=use_obb,
        bgr=True,
    )(labels)


# function for plotting the single image 
def plot_single_image(
    image_file: Path,
    label_file: Path,
    task: str,
    names: Optional[Dict[int, str]] = None,
    output_dir: Optional[Path] = None,
    masks_mode: str = "masks",
    verbose: bool = False,
):
    """
    Plot image annotations.

    Args:
        image_file: path to image file
        label_file: path to label file
        names: dict of class names
        task: task to visualize
        masks_mode: mask visualization mode, 'masks' or 'contours'
        verbose: verbose output
    """
    if not label_file.exists():
        LOGGER.warning(f"Skipping {image_file} as {label_file} does not exist")
        return

    if verbose:
        LOGGER.info(f"Plot image '{image_file}' and label '{label_file}'")

    use_keypoints = task in ["pose"]
    use_segments = task in ["segment"]
    use_obb = task in ["obb"]

    nkpt, ndim = data.get("kpt_shape", (0, 0))

    if use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
        raise ValueError(
            "'kpt_shape' in data.yaml missing or incorrect. "
            "Should be a list with [number of keypoints, number of dims "
            "(2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
        )

    try:
        labels = _get_labels(
            image_file=image_file,
            label_file=label_file,
            use_keypoints=use_keypoints,
            use_segments=use_segments,
            use_obb=use_obb,
            nkpt=nkpt,
            ndim=ndim,
            single_cls=False,
        )
    except ValueError as e:
        LOGGER.error(e)
        return

    fname = Path(image_file).name

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_name = output_dir / image_file.name
    else:
        save_name = None

    kwargs = dict()

    if "masks" in labels:
        kwargs["masks"] = labels["masks"]
        show_labels = False

    if "masks" not in labels or masks_mode == "masks":
        kwargs["bboxes"] = labels["bboxes"].reshape(-1, 4)
        show_labels = True

    if "keypoints" in labels:
        kwargs["kpts"] = labels["keypoints"]

    if names is not None:
        kwargs["names"] = names

    images = labels["img"][None]
    batch_idx = labels["batch_idx"].int()
    cls = labels["cls"].reshape(-1)

    try:
        plot_images(
            images=images,
            batch_idx=batch_idx,
            cls=cls,
            **kwargs,
            fname=fname,
            show_labels=show_labels,
            save=True,
            save_name=save_name,
            mode=masks_mode,
            use_thread=False,
        )
    except Exception as e:
        LOGGER.error(f"Error plotting image: {e}")
        return

# calling up the main function 
if __name__ == "__main__":
    parser = get_visualization_parser()
    args = parser.parse_args()

    names = defaultdict(lambda i: str(i))
    image_dirs = dict()

    if args.data is not None:
        try:
            if args.task == "classify":
                data = check_cls_dataset(args.data)
            elif str(args.data).split(".")[-1] in {
                "yaml",
                "yml",
            } or args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:
                data = check_det_dataset(args.data)

            for split in ("train", "val", "test"):
                image_dirs[split] = data.get(split, split)

            names = data["names"]

        except Exception as e:
            raise RuntimeError(
                emojis(f"Dataset '{clean_url(args.data)}' error ❌ {e}")
            ) from e

        LOGGER.info(f"Dataset: {data}")

    names = select_classes(
        names=names, class_names=args.class_names, classes=args.classes
    )
    LOGGER.info(f"Use {len(names)} classes: {dict(names)}")

    if args.image is not None:  # visualize single image
        plot_single_image(
            image_file=args.image,
            label_file=args.label,
            names=names,
            task=args.task,
            output_dir=args.output_dir,
            masks_mode=args.masks_mode,
            verbose=args.verbose,
        )
    else:  # all images of a dataset
        for split, image_dir in image_dirs.items():
            LOGGER.info(f"Visualizing '{split}' images in '{image_dir}'")

            image_files = list()
            for ext in IMG_FORMATS:
                image_files.extend(Path(image_dir).glob(f"*{ext}"))

            image_files = sorted(image_files)
            label_files = map(Path, img2label_paths(map(str, image_files)))

            if args.output_dir is None:
                output_dir = None
            else:
                output_dir = args.output_dir / split

            LOGGER.info(f"Found {len(image_files)} images")

            for image_file, label_file in zip(image_files, label_files):
                plot_single_image(
                    image_file=image_file,
                    label_file=label_file,
                    names=names,
                    task=args.task,
                    output_dir=output_dir,
                    masks_mode=args.masks_mode,
                    verbose=args.verbose,
                )
