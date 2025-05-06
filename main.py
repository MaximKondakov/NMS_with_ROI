import json
import argparse
from typing import List, Dict


def calculate_iou(box1: List[float],
                  box2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First bounding box in format [x_min, y_min, x_max, y_max].
        box2: Second bounding box in format [x_min, y_min, x_max, y_max].

    Returns:
        float: IoU value between 0 and 1.
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou


def get_roi_ids(bbox: List[float],
                roi_tlwh: List[List]) -> List[int]:
    """Get indices of ROIs that intersect with given bounding box.

    Args:
        bbox: Bounding box in format [top, left, bottom, right].
        roi_tlwh: List of ROIs, each in format [top, left, width, height].

    Returns:
        List[int]: Indices of intersecting ROIs.
    """
    x1, y1, x2, y2 = bbox
    roi_indices = []

    for i, (rx, ry, rw, rh) in enumerate(roi_tlwh):
        roi_x1, roi_y1 = rx, ry
        roi_x2, roi_y2 = rx + rw, ry + rh

        overlap_x1 = max(x1, roi_x1)
        overlap_y1 = max(y1, roi_y1)
        overlap_x2 = min(x2, roi_x2)
        overlap_y2 = min(y2, roi_y2)

        if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
            roi_indices.append(i)

    return roi_indices


def roi_nms(
            detections: List[Dict],
            roi_tlwh: List[List],
            iou_threshold: float = 0.5,
        ) -> List[Dict]:
    """Perform Non-Maximum Suppression (NMS) considering Regions of Interest (ROIs).

    Args:
        detections: List of detection dictionaries.
        roi_tlwh: List of ROIs, each in format [x, y, width, height].
        iou_threshold: IoU threshold for suppression (default: 0.5).

    Returns:
        List[Dict]: Filtered detections after NMS.
    """
    for det in detections:
        det["roi_ids"] = get_roi_ids(det["bbox"], roi_tlwh)

    detections = sorted(detections, key=lambda x: x["score"], reverse=True)
    selected = []

    while detections:
        best = detections.pop(0)
        selected.append(best)

        remaining_detections = []
        for det in detections:
            iou = calculate_iou(best["bbox"], det["bbox"])

            if iou < iou_threshold:
                remaining_detections.append(det)
                continue

            common_rois = set(best["roi_ids"]) & set(det["roi_ids"])
            if not common_rois:
                remaining_detections.append(det)
                continue

        detections = remaining_detections

    return selected


def wonderful_player_detections(player_data: List[Dict]) -> List[Dict]:
    """Filter player detections using ROI-based NMS.

    Args:
        player_data: List of player detection dictionaries.

    Returns:
        List[Dict]: Filtered player detections.
    """
    roi_tlwh = [
        [36, 82, 1776, 1776],
        [1705, 80, 1532, 1532],
        [3126, 168, 1280, 1280],
        [4310, 80, 1532, 1532],
        [5774, 82, 1776, 1776],
        [1036, 1322, 5464, 1500],
        [0, 2852, 1938, 1938],
        [1848, 2852, 1938, 1938],
        [3786, 2852, 1938, 1938],
        [5638, 2852, 1938, 1938]
    ]

    filtered_player_data = []

    # Grouped frames
    frame_to_detections = {}
    for detection in player_data:
        frame = detection['frame']
        if frame not in frame_to_detections:
            frame_to_detections[frame] = []
        frame_to_detections[frame].append(detection)

    # For each frame we use NMS
    for frame, detections in frame_to_detections.items():
        filtered_detections = roi_nms(
            detections,
            roi_tlwh,
            iou_threshold=0.5,
        )
        filtered_player_data.extend(filtered_detections)

    return filtered_player_data


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process player detections from JSON file.')
    parser.add_argument('--input', type=str, required=True, help='Path to input JSON')
    return parser.parse_args()


def main():
    """Main function to load data and apply player detection filtering."""
    args = parse_arguments()
    with open(args.input, 'r') as input_file:
        data = json.load(input_file)
    filtered_player_data = wonderful_player_detections(data)
    print('complete')


if __name__ == '__main__':
    main()
