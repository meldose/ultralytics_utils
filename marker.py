#!/usr/bin/env python3
import argparse
import os
import cv2
import yaml
import numpy as np
import open3d as o3d

def load_camera_params(yaml_file):
    """Load camera intrinsics from a ROS/OpenCV‐style YAML."""
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    K = np.array(data['camera_matrix']['data']).reshape((3,3))
    dist = np.array(data['distortion_coefficients']['data'])
    return K, dist

def detect_markers(img, aruco_dict, params):
    """Detect ArUco markers and draw their 2D boundary."""
    corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=params)
    vis = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)
    return corners, ids, vis

def estimate_pose(vis, corners, K, dist, marker_length):
    """Estimate and draw each marker's 6‐DoF pose as colored axes."""
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, marker_length, K, dist
    )
    for r, t in zip(rvecs, tvecs):
        cv2.aruco.drawAxis(vis, K, dist, r, t, marker_length * 0.5)
    return rvecs, tvecs, vis

def project_mesh_mask(mesh_file, K, rvec, tvec, img_shape):
    """Project the 3D mesh into the image plane and rasterize its silhouette."""
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    verts = np.asarray(mesh.vertices)

    # Rotate + translate into camera coords
    R, _ = cv2.Rodrigues(rvec)
    cam_pts = (R @ verts.T + tvec.T).T

    # Project to 2D
    pts2d, _ = cv2.projectPoints(verts, rvec, tvec, K, None)
    pts2d = pts2d.reshape(-1, 2).astype(int)

    # Build convex hull and draw filled mask
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(pts2d)
    cv2.drawContours(mask, [hull], -1, 255, thickness=-1)
    return mask

def process_image(path, args, aruco_dict, detector_params, K=None, dist=None):
    img = cv2.imread(path)
    basename = os.path.splitext(os.path.basename(path))[0]

    # 1) Detect markers
    corners, ids, vis = detect_markers(img, aruco_dict, detector_params)

    # 2) If intrinsics provided, estimate pose & optionally project mesh
    if K is not None and len(corners) > 0:
        rvecs, tvecs, vis = estimate_pose(vis, corners, K, dist, args.marker_length)

        # Use the first marker's pose to project your mesh
        if args.mesh and len(rvecs) > 0:
            mask = project_mesh_mask(
                args.mesh, K, rvecs[0], tvecs[0], img.shape
            )
            cv2.imwrite(os.path.join(args.output, f"{basename}_mask.png"), mask)

    # 3) Save visualization
    out_path = os.path.join(args.output, f"{basename}_annotated.jpg")
    cv2.imwrite(out_path, vis)
    print(f"[+] Saved: {out_path}")

def main():
    p = argparse.ArgumentParser(
        description="ArUco Marker Detection → (Optional) Pose → (Optional) Mesh Projection"
    )
    p.add_argument("--input",      required=True, help="Image file or directory")
    p.add_argument("--output",     required=True, help="Output directory")
    p.add_argument("--camera_params",
                   help="YAML file with camera_matrix & distortion_coefficients")
    p.add_argument("--marker_length", type=float, default=0.10,
                   help="Marker side length in meters")
    p.add_argument("--mesh", help="Path to 3D mesh (OBJ/PLY) for silhouette projection")
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Prepare ArUco detector
    aruco_dict     = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    detector_params = cv2.aruco.DetectorParameters_create()

    # Load intrinsics if given
    K = dist = None
    if args.camera_params:
        K, dist = load_camera_params(args.camera_params)

    # Gather image paths
    if os.path.isdir(args.input):
        imgs = [
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    else:
        imgs = [args.input]

    # Process each
    for img_path in sorted(imgs):
        process_image(img_path, args, aruco_dict, detector_params, K, dist)

if __name__ == "__main__":
    main()
