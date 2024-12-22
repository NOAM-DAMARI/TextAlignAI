import cv2
import pytesseract
import numpy as np

def straighten_text(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines using Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    angle = 0

    # Calculate average angle of the lines
    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle = (theta - np.pi / 2) * (180 / np.pi)
            angles.append(angle)
        angle = np.mean(angles)

    # Rotate the image
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Save the result
    cv2.imwrite(output_path, rotated)
    print(f"Straightened image saved to {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Straighten text in an image.")
    parser.add_argument("input", help="Path to the input image")
    parser.add_argument("output", help="Path to save the straightened image")
    args = parser.parse_args()

    straighten_text(args.input, args.output)
