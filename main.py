import cv2
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import os

@dataclass
class Point:
    x: float
    y: float

class Line:
    def __init__(self, end1: Point = None, end2: Point = None):
        self.type = "L"
        self.x1 = end1.x if end1 else 0
        self.y1 = end1.y if end1 else 0
        self.x2 = end2.x if end2 else 0
        self.y2 = end2.y if end2 else 0
    
    def get_eq(self) -> str:
        return f"\\left(\\left(1-t\\right)*{self.x1}+t*{self.x2},\\left(1-t\\right)*{-self.y1}+t*{-self.y2}\\right)"

class Bezier:
    def __init__(self, end1: Point = None, ctrl1: Point = None, 
                 ctrl2: Point = None, end2: Point = None):
        self.type = "C"
        self.x1 = end1.x if end1 else 0
        self.y1 = end1.y if end1 else 0
        self.cx1 = ctrl1.x if ctrl1 else 0
        self.cy1 = ctrl1.y if ctrl1 else 0
        self.cx2 = ctrl2.x if ctrl2 else 0
        self.cy2 = ctrl2.y if ctrl2 else 0
        self.x2 = end2.x if end2 else 0
        self.y2 = end2.y if end2 else 0
    
    def get_eq(self) -> str:
        return f"\\left(\\left(1-t\\right)^{{3}}*{self.x1}+3t\\left(1-t\\right)^{{2}}*{self.cx1}+" \
               f"3t^{{2}}\\left(1-t\\right)*{self.cx2}+t^{{3}}*{self.x2}\\ ,\\left(1-t\\right)^{{3}}*" \
               f"{-self.y1}+3t\\left(1-t\\right)^{{2}}*{-self.cy1}+3t^{{2}}\\left(1-t\\right)*" \
               f"{-self.cy2}+t^{{3}}*{-self.y2}\\right)"

def extract_contours(binary: np.ndarray) -> List[np.ndarray]:
    """Extract contours from a binary image."""
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def fit_curves(contours: List[np.ndarray]) -> List:
    """Fit lines or Bezier curves to contours."""
    curves = []
    for contour in contours:
        # Fit a line or Bezier to every segment of the contour
        for i in range(len(contour) - 1):
            x1, y1 = contour[i][0]
            x2, y2 = contour[i + 1][0]
            if i + 2 < len(contour):  # Use next point for Bezier fitting
                x3, y3 = contour[i + 2][0]
                bezier = Bezier(Point(x1, y1), Point((x1 + x2) / 2, (y1 + y2) / 2),
                                Point((x2 + x3) / 2, (y2 + y3) / 2), Point(x3, y3))
                curves.append(bezier)
            else:
                line = Line(Point(x1, y1), Point(x2, y2))
                curves.append(line)
    return curves

def plot_results(original: np.ndarray, edges: np.ndarray, curves: List, output_path: str) -> None:
    """Visualize and save the original, edge detection, and curves."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axs[0].imshow(original, cmap='gray')
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    # Edges
    axs[1].imshow(edges, cmap='gray')
    axs[1].set_title("Edge Detection")
    axs[1].axis("off")

    # Curves
    canvas = np.zeros_like(original)
    for curve in curves:
        if curve.type == "L":
            cv2.line(canvas, 
                     (int(curve.x1), int(curve.y1)), 
                     (int(curve.x2), int(curve.y2)), 
                     255, 1)
        elif curve.type == "C":
            pts = np.array([(curve.x1, curve.y1), (curve.cx1, curve.cy1),
                            (curve.cx2, curve.cy2), (curve.x2, curve.y2)], np.int32)
            cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, 255, 1)

    axs[2].imshow(canvas, cmap='gray')
    axs[2].set_title("Curves")
    axs[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_image(image_path: str, output_dir: str = "output"):
    """Process image: Detect edges, extract contours, fit curves, and save results."""
    os.makedirs(output_dir, exist_ok=True)

    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    # Edge detection
    edges = cv2.Canny(img, 100, 200)

    # Contour extraction
    contours = extract_contours(edges)

    # Fit curves
    curves = fit_curves(contours)

    # Save curve equations
    equations = [curve.get_eq() for curve in curves]
    with open(os.path.join(output_dir, "equations.txt"), "w") as f:
        f.write("\n".join(equations))

    # Plot and save visualization
    plot_results(img, edges, curves, os.path.join(output_dir, "results.png"))

    return curves, equations

if __name__ == "__main__":
    input_image = "a.jpg"  # Replace with your image path
    output_directory = "output"

    try:
        curves, equations = process_image(input_image, output_directory)
        print(f"Processing completed. Results saved to {output_directory}.")
    except Exception as e:
        print(f"Error: {e}")
