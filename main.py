import cv2
import numpy as np
import json
import os


def segment_plots(image_path, output_img, output_json):

    image = cv2.imread(image_path)
    if image is None:
        print("Failed to read:", image_path)
        return

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ---- 1. Slightly relaxed white threshold ----
    _, white_mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)

    # ---- 2. Morphological cleanup ----
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    # ---- 3. Connected components ----
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        white_mask, connectivity=8
    )

    plots = []

    for i in range(1, num_labels):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # ---- Basic filtering ----
        if area < 2500:
            continue

        if bh < 20:
            continue

        # ---- Remove wide thin divider ----
        aspect_ratio = bw / float(bh)
        if aspect_ratio > 6:
            continue

        # Remove border touching
        if x == 0 or y == 0 or (x + bw) >= w or (y + bh) >= h:
            continue

        component_mask = np.zeros((h, w), dtype=np.uint8)
        component_mask[labels == i] = 255

        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        cnt = contours[0]

        perimeter = cv2.arcLength(cnt, True)
        epsilon = 0.005 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        polygon_px = []
        polygon_norm = []

        for pt in approx:
            px, py = pt[0]
            polygon_px.append([float(px), float(py)])
            polygon_norm.append([
                round(px / w, 6),
                round(py / h, 6)
            ])

        cx, cy = int(centroids[i][0]), int(centroids[i][1])

        plots.append({
            "polygon_px": polygon_px,
            "polygon_norm": polygon_norm,
            "centroid_px": [cx, cy],
            "contour_area_px": float(area),
            "bbox_px": [int(x), int(y), int(x + bw), int(y + bh)],
            "plot_number_info": None
        })

    # ---- Row-wise sorting ----
    row_tolerance = 50

    plots = sorted(
        plots,
        key=lambda p: (
            p["centroid_px"][1] // row_tolerance,
            p["centroid_px"][0]
        )
    )

    # ---- Assign IDs and draw ----
    for idx, plot in enumerate(plots):

        plot["id_auto"] = idx + 1

        pts = np.array(plot["polygon_px"], dtype=np.int32)
        cv2.polylines(image, [pts], True, (0, 255, 0), 2)

        cx, cy = plot["centroid_px"]

        cv2.putText(
            image,
            str(plot["id_auto"]),
            (cx - 10, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    cv2.imwrite(output_img, image)

    output_data = {
        "image_path": image_path,
        "image_size": [w, h],
        "plots": plots
    }

    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"{os.path.basename(image_path)} -> Detected {len(plots)} plots")


def process_directory(input_dir):

    output_dir = os.path.join(input_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):

        if file.lower().endswith((".png", ".jpg", ".jpeg")):

            input_path = os.path.join(input_dir, file)
            name = os.path.splitext(file)[0]

            output_img = os.path.join(output_dir, f"{name}_detected.jpg")
            output_json = os.path.join(output_dir, f"{name}.json")

            print("Processing:", file)

            segment_plots(input_path, output_img, output_json)

    print("Done.")


if __name__ == "__main__":
    process_directory("images")
