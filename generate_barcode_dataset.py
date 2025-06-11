import argparse
import multiprocessing as mp
from pathlib import Path
import random
import string
from typing import List

import cv2
import numpy as np
from random_words import RandomWords
from shapely.geometry import Polygon
import treepoem
from tqdm import tqdm
from wordcloud import WordCloud
import yaml

# Configuration
BARCODE_TYPES = {
    "ean13": {
        "size": 12,
        "H": 146,
        "W": 190,
        "class_id": 0,
        "fixed_size": True
    },
    "qrcode": {
        "min_size": 30,
        "max_size": 100,
        "H": 102,
        "W": 102,
        "class_id": 1,
        "fixed_size": False
    },
    'pdf417': {
        'min_size': 30,
        "max_size": 100,
        "H": 62,
        "W": 242,
        "class_id": 2,
        "fixed_size": False
    },
    "azteccode": {
        "min_size": 30,
        "max_size": 100,
        "H": 102,
        "W": 102,
        "class_id": 3,
        "fixed_size": False
    },
    'azteccodecompact': {
        'min_size': 10,
        "max_size": 50,
        "H": 102,
        "W": 102,
        "class_id": 4,
        "fixed_size": False
    },
    "aztecrune": {
        "size": 1,
        "H": 102,
        "W": 102,
        "class_id": 5,
        "fixed_size": True
    },
}


def calculate_obb_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate IoU between two oriented bounding boxes
    
    Args:
        box1: [x1,y1,x2,y2,x3,y3,x4,y4] first box corners
        box2: [x1,y1,x2,y2,x3,y3,x4,y4] second box corners
    Returns:
        IoU score
    """
    # Convert boxes to polygons
    poly1 = Polygon(np.array(box1).reshape(4, 2))
    poly2 = Polygon(np.array(box2).reshape(4, 2))

    # Check for valid polygons
    if not (poly1.is_valid and poly2.is_valid):
        print("Invalid polygons")
        return 0.0

    # Calculate intersection and union
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area

    # Calculate IoU
    iou = intersection / union if union > 0 else 0.0

    return float(iou)


def calculate_barcode_size(
        height: int,
        width: int,
        num_barcodes: int
    ):
    """Calculate appropriate barcode size based on image size and
    number of barcodes"""
    # Base size calculation
    total_area = height * width
    # Using 1/4 of equal distribution
    area_per_barcode = total_area / (num_barcodes * 4)

    # Calculate base dimension (square root of area)
    base_dim = int(np.sqrt(area_per_barcode))

    # Add some randomness (±20%)
    random_factor = random.uniform(0.8, 1.2)
    base_dim = int(base_dim * random_factor)

    # Ensure minimum and maximum sizes
    size = min(height, width)
    min_dim = size // 20  # Minimum 5% of image size
    max_dim = size // 3   # Maximum 33% of image size

    return np.clip(base_dim, min_dim, max_dim)


def create_barcode(target_size, code_type: str = "random"):
    """Create a barcode with dynamic sizing"""
    if code_type == "random":
        code_type = random.choice(list(BARCODE_TYPES.keys()))

    # Get class_id for the barcode type
    class_id = BARCODE_TYPES[code_type]['class_id']

    # Determine data size based on barcode type
    if BARCODE_TYPES[code_type]['fixed_size']:
        data_size = BARCODE_TYPES[code_type]['size']
    else:
        data_size = random.randint(
            BARCODE_TYPES[code_type]['min_size'],
            BARCODE_TYPES[code_type]['max_size']
        )

    # Generate data string
    if code_type in ["qrcode", "pdf417", "azteccode", "azteccodecompact"]:
        # For variable-size codes, use a mix of digits and letters
        chars = string.ascii_letters + string.digits
        data = ''.join(random.choice(chars) for _ in range(data_size))
    else:
        # For fixed-size codes (like EAN13), use only digits
        data = ''.join(random.choice(string.digits) for _ in range(data_size))

    # Generate barcode
    barcode = np.array(
        treepoem.generate_barcode(
            barcode_type=code_type, data=data
        ).convert('1')
    ).astype(np.uint8)

    # Ensure barcode stripes are black (0) and background is white (255)
    barcode = 255 - (barcode * 255)

    # Resize barcode to target size
    barcode = cv2.resize(barcode, (target_size, target_size))

    return barcode, class_id


def generate_random_text():
    rw = RandomWords()
    # Reduce number of words significantly
    words = rw.random_words(count=10)  # Reduced from 50
    numbers = [str(random.randint(1000, 9999999)) for _ in range(5)]

    return ' '.join(words + numbers)


def rotate_image(
    image: np.ndarray,
    angle: float,
    border_value: tuple[int, int, int] = (0, 0, 0)
):
    height, width = image.shape[:2]
    center = np.array([width // 2, height // 2])

    corners = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ]) - center
    corners = np.hstack([corners, np.ones((4, 1))])

    R = cv2.getRotationMatrix2D(tuple(center.tolist()), angle, 1.0)

    abs_cos = abs(R[0, 0])
    abs_sin = abs(R[0, 1])
    
    rotated_w = int(height * abs_sin + width * abs_cos)
    rotated_h = int(height * abs_cos + width * abs_sin)

    rotated_center = np.array([rotated_w / 2, rotated_h / 2])
    
    R[0, 2] += rotated_center[0] - center[0]
    R[1, 2] += rotated_center[1] - center[1]

    rotated_corners = (R @ corners.T).T
    min_x = np.min(rotated_corners, axis=0)[0]
    min_y = np.min(rotated_corners, axis=0)[1]
    rotated_corners -= np.array([min_x, min_y])
    rotated_corners = np.round(rotated_corners[:, :2]).astype(np.int32)

    rotated_image = cv2.warpAffine(
        image,
        R,
        (rotated_w, rotated_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )

    mask = np.ones_like(image) * 255
    rotated_mask = cv2.warpAffine(
        mask,
        R,
        (rotated_w, rotated_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    rotated_mask = rotated_mask > 128

    return rotated_image, rotated_mask, rotated_corners


def generate_random_placement(
    width: int,
    height: int,
    patch_width: int,
    patch_height: int,
) -> tuple[int, int]:
    """Generate random placement of barcode in image"""
    x = random.randint(0, width - patch_width)
    y = random.randint(0, height - patch_height)

    return x, y


def generate_single_image(
    height: int=640,
    width: int=640,
    max_barcodes_per_image: int=10,
    max_attempts: int=100,
    max_iou: float=0.0,
):
    num_barcodes = random.randint(1, max_barcodes_per_image)

    # Calculate base barcode size for this image
    barcode_size = calculate_barcode_size(
        height=height,
        width=width,
        num_barcodes=num_barcodes
    )

    background_color = tuple(random.randint(220, 255) for _ in range(3))

    size = min(width, height)

    background = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        min_font_size=size//30,  # Increased minimum font size
        max_font_size=size//10,  # Increased maximum font size
        max_words=15,            # Reduced from size//10
        prefer_horizontal=0.6    # 60% horizontal text
    ).generate(generate_random_text()).to_array()

    # List to store YOLO format boxes for output
    bboxes = list()

    for idx in range(num_barcodes):
        # Create barcode with calculated size and get class_id
        barcode, class_id = create_barcode(target_size=barcode_size)
        barcode_rgb = np.stack([barcode] * 3, axis=-1)

        angle = random.uniform(-90, 90)
        rotated_barcode, mask, corners = rotate_image(
            barcode_rgb, angle, border_value=(0, 255, 0)
        )

        rotated_h, rotated_w = rotated_barcode.shape[:2]

        # Try to place barcode without overlap
        success = False

        for _ in range(max_attempts):
            x, y = generate_random_placement(
                width=width,
                height=height,
                patch_width=rotated_w,
                patch_height=rotated_h,
            )

            # Convert to Ultralytics OBB format
            bbox = (corners + np.array([x, y])) / np.array([width, height])

            # Check if all coordinates lie within image bounds
            if not all(0 <= x <= 1 and 0 <= y <= 1 for x, y in bbox):
                continue

            bbox = [class_id] + bbox.ravel().tolist()

            if any(
                calculate_obb_iou(bbox[1:], other_bbox[1:]) > max_iou
                for other_bbox in bboxes
            ):
                continue

            bboxes.append(bbox)

            roi = background[y:y+rotated_h, x:x+rotated_w]
            roi[mask] = rotated_barcode[mask]
            background[y:y+rotated_h, x:x+rotated_w] = roi

            success = True

            break
                
        if not success:
            continue  # Skip this barcode if we couldn't place it

    return background, bboxes


def generate_sample(task: tuple[int, Path]):
    idx, output_dir = task

    frame_id = f"{idx:06d}"

    image, bboxes = generate_single_image()

    # Save image
    image_path = output_dir / "images" / f"{frame_id}.png"
    cv2.imwrite(image_path, image)

    # Save labels
    label_path = output_dir / "labels" / f"{frame_id}.txt"

    with open(label_path, 'w+') as f:
        for bbox in bboxes:
            f.write(' '.join(map(str, bbox)) + '\n')


def main(output_dir: Path, num_samples: int, num_processes: int = 1):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)

    dataset_name = output_dir.name
    # Create config file
    cfg = {
        "path": output_dir.absolute().as_posix(),
        "nc": len(BARCODE_TYPES),
        "names": list(BARCODE_TYPES.keys())
    }

    with open(output_dir / f"{dataset_name}.yaml", "w") as f:
        yaml.dump(cfg, f)

    # Generate samples
    tasks = [(idx, output_dir) for idx in range(num_samples)]

    with mp.Pool(processes=num_processes) as pool:
        for _ in tqdm(pool.imap(generate_sample, tasks), total=num_samples):
            pass

    print(f"Generated {num_samples} samples in YOLO format")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', "-o", type=Path, default="./dataset")
    parser.add_argument('--num-samples', "-n", type=int, default=5000)
    parser.add_argument('--num-processes', "-p", type=int, default=10)
    args = parser.parse_args()

    main(**vars(args))
