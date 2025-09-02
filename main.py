import sys
import pathlib

from scipy.signal import convolve2d
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class LetterPixel:
    LETTER_DENSE = ".,-~:;=*#@"
    def get(value):
        assert 0 <= value < 10, "Value not in range [0-9]"
        return LetterPixel.LETTER_DENSE[value]

def luminance(arr: np.ndarray) -> np.ndarray:
    """
    arr: np.ndarray of shape h, w, 3
    """
    rgb = np.array([0.2126, 0.7152, 0.0722])
    light = (arr * rgb).sum(-1)
    light = (light - light.min()) / (light.max() - light.min())
    return light

def line(arr: np.ndarray) -> np.ndarray:
    kernelup = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernelleft = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    linesup = convolve2d(arr, kernelup, mode="same")
    linesleft = convolve2d(arr, kernelleft, mode="same")

    linesup = (linesup - linesup.min()) / (linesup.max() - linesup.min())
    linesleft = (linesleft - linesleft.min()) / (linesleft.max() - linesleft.min())

    outup = np.histogram(linesup, bins=20)
    maxup = outup[0].argmax()
    maskup = (linesup < outup[1][max(maxup - 1, 0)]) + (linesup > outup[1][min(maxup + 1, len(outup[0]))])

    outleft = np.histogram(linesleft, bins=20)
    maxleft = outleft[0].argmax()
    maskleft = (linesleft < outleft[1][max(maxleft - 1, 0)]) + (linesleft > outleft[1][min(maxleft + 1, len(outleft[0]))])
    mask = maskleft + maskup
    
    lines = linesleft + linesup
    lines[~mask] = 0

    mask = (linesleft > 0) * (linesup > 0)
    lines[mask] /= 2
    return lines

def main():
    assert len(sys.argv) == 2, "Need argument for image [ python main.py <image_path> ]."

    image_path = pathlib.Path(sys.argv[1]).expanduser().absolute()
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)


    light = luminance(image)
    light *= (10 - 10e-6)
    light = np.floor(light).astype(np.int32)
    lines = line(light)
    

    out = np.vectorize(LetterPixel.get)(light)

    art = ""
    for row in out:
        art += "".join(row) + "\n"
    
    with open("output.txt", "w") as f:
        f.write(art)

if __name__ == "__main__":
    main()