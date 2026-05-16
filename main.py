#!/usr/bin/env python3
"""
Zeon Assignment 1: Hybrid Lid Orientation Estimation Pipeline
Author: Jaywardhan Raghu (23B0737)

Provides a clean, standalone production CLI script to evaluate any arbitrary 
target image using the trained dual-YOLOv11 model pipeline.
"""

import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from inference_sdk import InferenceHTTPClient

# ==============================================================================
# DEFAULT RUNTIME CONFIGURATION
# ==============================================================================
API_KEY = "vof93SgrEEFl6QUIp6BI"
SEGMENTATION_MODEL_ID = "zeon-9tvbv/1"
KEYPOINT_MODEL_ID = "zeon-2/2"

PADDING = 15
ANGLE_THRESHOLD = 10.0


def process_hybrid_pipeline(image_path, api_key=API_KEY):
    """
    Executes the complete dual-model computer vision pipeline on a target image:
    1. YOLOv11 Segmentation for contour mask extraction.
    2. Parallel Ellipse Fitting and Principal Component Analysis (PCA).
    3. Crop-based YOLOv11 Keypoint Detection (Hinge/Tab coordinates).
    4. 180° vector ambiguity resolution.
    5. Dynamic threshold switching logic to establish final orientation vector.
    """
    if not os.path.exists(image_path):
        print(f"[-] Error: Target image path '{image_path}' does not exist.")
        return

    # Load Source Matrix
    image = cv2.imread(image_path)
    output_display = image.copy()
    print(f"[+] Loaded: {image_path} ({image.shape[1]}x{image.shape[0]})")

    # Initialize Client Context
    try:
        client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=api_key
        )
    except Exception as err:
        print(f"[-] Client initialization failed: {err}")
        return

    # Run Primary Segmentation Request
    print("[+] Querying YOLOv11 Segmentation Model...")
    try:
        seg_result = client.infer(image_path, model_id=SEGMENTATION_MODEL_ID)
        predictions = seg_result["predictions"]
    except Exception as err:
        print(f"[-] Segmentation inference failure: {err}")
        return

    print(f"[+] Detected lids in workspace: {len(predictions)}")

    # Iterate over individual object instances
    for idx, pred in enumerate(predictions):
        try:
            # ------------------------------------------------------------------
            # Contour Extraction
            # ------------------------------------------------------------------
            points = pred["points"]
            contour = np.array(
                [[int(p["x"]), int(p["y"])] for p in points],
                dtype=np.int32
            )

            if len(contour) < 5:
                continue

            # ------------------------------------------------------------------
            # PCA Convex Hull Smoothing Setup
            # ------------------------------------------------------------------
            contour_pca = cv2.convexHull(contour)
            epsilon = 0.01 * cv2.arcLength(contour_pca, True)
            contour_pca = cv2.approxPolyDP(contour_pca, epsilon, True)

            # ------------------------------------------------------------------
            # Standalone Pipeline 1: Ellipse Fit (High Bias / Low Robustness)
            # ------------------------------------------------------------------
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (axis1, axis2), raw_angle = ellipse
            cx, cy = int(cx), int(cy)

            if axis1 >= axis2:
                ellipse_major_axis = axis1
                ellipse_minor_axis = axis2
                ellipse_major_angle = raw_angle
            else:
                ellipse_major_axis = axis2
                ellipse_minor_axis = axis1
                ellipse_major_angle = raw_angle + 90.0

            ellipse_major_angle %= 360.0
            theta_e = np.deg2rad(ellipse_major_angle)
            ellipse_axis_vec = np.array([np.cos(theta_e), -np.sin(theta_e)])

            # ------------------------------------------------------------------
            # Standalone Pipeline 2: PCA Axis Computation (Low Bias / High Robustness)
            # ------------------------------------------------------------------
            pts = contour_pca.reshape(-1, 2).astype(np.float32)
            _, eigenvectors = cv2.PCACompute(pts, mean=None)
            principal_vec = eigenvectors[0]

            vx, vy = principal_vec[0], -principal_vec[1]
            pca_major_angle = np.degrees(np.arctan2(vy, vx)) % 360.0

            theta_p = np.deg2rad(pca_major_angle)
            pca_axis_vec = np.array([np.cos(theta_p), np.sin(theta_p)])

            # ------------------------------------------------------------------
            # Landmark Keypoint Inference Pipeline
            # ------------------------------------------------------------------
            x, y, w, h = cv2.boundingRect(contour)
            x1 = max(0, x - PADDING)
            y1 = max(0, y - PADDING)
            x2 = min(image.shape[1], x + w + PADDING)
            y2 = min(image.shape[0], y + h + PADDING)

            crop = image[y1:y2, x1:x2]
            temp_crop_path = f"temp_crop_instance_{idx}.jpg"
            cv2.imwrite(temp_crop_path, crop)

            kp_result = client.infer(temp_crop_path, model_id=KEYPOINT_MODEL_ID)
            kp_predictions = kp_result["predictions"]

            # Local cleanup of written runtime crop assets
            if os.path.exists(temp_crop_path):
                os.remove(temp_crop_path)

            if len(kp_predictions) == 0:
                print(f"[-] Keypoint skip on Lid {idx}: Model returned empty array.")
                continue

            kp = kp_predictions[0]
            keypoints = kp["keypoints"]

            hinge, tab = None, None
            for k in keypoints:
                cls_name = k["class"].lower()
                px = k["x"] + x1
                py = k["y"] + y1

                if "hinge" in cls_name:
                    hinge = np.array([px, py])
                elif "tab" in cls_name:
                    tab = np.array([px, py])

            if hinge is None or tab is None:
                continue

            # Compute Disambiguation Normal Vector
            kp_vec = tab - hinge
            kp_norm = np.linalg.norm(kp_vec)
            if kp_norm < 1e-6:
                continue
            kp_vec /= kp_norm
            kp_vec[1] *= -1.0  # Image coordinate conversion check

            # ------------------------------------------------------------------
            # Disambiguate Ellipse Vector 180° Ambiguity
            # ------------------------------------------------------------------
            if np.dot(ellipse_axis_vec, kp_vec) >= np.dot(-ellipse_axis_vec, kp_vec):
                ellipse_final_vec = ellipse_axis_vec
            else:
                ellipse_final_vec = -ellipse_axis_vec

            ellipse_true_angle = np.degrees(np.arctan2(-ellipse_final_vec[1], ellipse_final_vec[0])) % 360.0
            ellipse_true_angle = (360.0 - ellipse_true_angle) % 360.0

            # ------------------------------------------------------------------
            # Disambiguate PCA Vector 180° Ambiguity
            # ------------------------------------------------------------------
            if np.dot(pca_axis_vec, kp_vec) >= np.dot(-pca_axis_vec, kp_vec):
                pca_final_vec = pca_axis_vec
            else:
                pca_final_vec = -pca_axis_vec

            pca_true_angle = np.degrees(np.arctan2(pca_final_vec[1], pca_final_vec[0])) % 360.0

            # ------------------------------------------------------------------
            # Dynamic Hybrid Selection Rule Evaluation
            # ------------------------------------------------------------------
            dot_product = np.clip(np.dot(ellipse_final_vec, pca_final_vec), -1.0, 1.0)
            disagreement = np.degrees(np.arccos(abs(dot_product)))

            if disagreement > ANGLE_THRESHOLD:
                selected_method = "PCA"
                final_vec = pca_final_vec
                true_angle = pca_true_angle
                ellipse_draw_angle = (-pca_major_angle) % 360.0
            else:
                selected_method = "ELLIPSE"
                final_vec = ellipse_final_vec
                true_angle = ellipse_true_angle
                ellipse_draw_angle = ellipse_major_angle

            # ------------------------------------------------------------------
            # Render Engineering Annotations
            # ------------------------------------------------------------------
            # 1. Overlay green segmentation profile boundary
            cv2.drawContours(output_display, [contour], -1, (0, 255, 0), 1)

            # 2. Draw yellow baseline geometric ellipse contour
            corrected_ellipse = ((cx, cy), (ellipse_major_axis, ellipse_minor_axis), ellipse_draw_angle)
            cv2.ellipse(output_display, corrected_ellipse, (0, 255, 255), 1)

            # 3. Draw red resolved directional output vector
            arrow_length = int(ellipse_major_axis * 0.45)
            end_x = int(cx + arrow_length * final_vec[0])
            end_y = int(cy - arrow_length * final_vec[1])
            cv2.arrowedLine(output_display, (cx, cy), (end_x, end_y), (0, 0, 255), 2, tipLength=0.25)

            # 4. Core localized point tag
            cv2.circle(output_display, (cx, cy), 3, (255, 0, 255), -1)

            # 5. Output angle text string positioning
            cv2.putText(
                output_display, f"{true_angle:.1f}", (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
            )

            print(f" -> Lid {idx:02d}: Ellipse={ellipse_true_angle:5.1f}° | PCA={pca_true_angle:5.1f}° | "
                  f"Chosen={true_angle:5.1f}° [{selected_method}] Delta={disagreement:5.2f}°")

        except Exception as err:
            print(f"[-] Execution breakdown on indexed object {idx}: {err}")

    # Output Rendering Sequences
    output_filename = "hybrid_evaluation_output.png"
    cv2.imwrite(output_filename, output_display)
    print(f"\n[+] Processing pipeline execution halt clean.")
    print(f"[+] Overlay rendering frame successfully written locally to: {output_filename}")

    # Display plot via headless check fallback
    rgb_frame = cv2.cvtColor(output_display, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_frame)
    plt.title(f"Hybrid Pipeline Evaluation Output\nSource: {os.path.basename(image_path)}")
    plt.axis("off")
    print("[+] Plot rendering execution complete. Closing local pipeline runtime.")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zeon Robotics Lid Positioning & Orientation CLI Pipeline Asset"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="images/originals/46374b36-color.png",
        help="Relative or explicit absolute string path targeting an input lid image asset."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=API_KEY,
        help="Roboflow Private Ingestion Platform Identification Token key string."
    )
    args = parser.parse_args()

    process_hybrid_pipeline(image_path=args.image, api_key=args.api_key)
