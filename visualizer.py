from PIL import Image, ImageDraw, ImageFont
import numpy as np


class Visualizer:

    def visualize(txt, colors: np.ndarray, fontsize=10):
        txt = txt
        fontsize = fontsize
        w = len(txt[0])
        h = len(txt)

        im = Image.fromarray(
            np.zeros(shape=(int(h * fontsize), int(w * fontsize), 3))
            .astype(np.uint8)
        )

        font = ImageFont.load_default(size=fontsize)
        draw = ImageDraw.Draw(im)

        for i, row in enumerate(txt):
            for j, letter in enumerate(row):
                draw.text(
                    xy=(j * fontsize, i * fontsize),
                    text=letter,
                    font=font,
                    fill=tuple((x.item() for x in colors[i, j]))
                )

        return im

# arr = np.zeros(shape=(1920, 1080, 3)).astype(np.uint8)
# im = Image.fromarray(arr)

# font = ImageFont.load_default(size=10)
# draw = ImageDraw.Draw(im)

# draw.text(xy=(0,0), text="text", font=font, fill=(255, 0, 0))
