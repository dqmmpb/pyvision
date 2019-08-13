import numpy

from PIL import Image, ImageDraw
import itertools
import random
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator


logger = logging.getLogger("vision.visualize")

defaultwidth = 1
colors = ["#FF00FF",
          "#FF0000",
          "#FF8000",
          "#FFD100",
          "#008000",
          "#0080FF",
          "#0000FF",
          "#000080",
          "#800080"]

# convert fig to img
def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    X = numpy.frombuffer(s, numpy.uint8).reshape((height, width, 4))
    im = Image.frombytes("RGBA", (width, height), s)
    return im


def highlight_box(image, box, color = colors[0], width = defaultwidth,
                  font = None):
    """
    Highlights the bounding box on the given image.
    """
    # draw = ImageDraw.Draw(image)
    fig, ax = plt.subplots(figsize=(image.width / 100.0, image.height / 100.0))

    # remove margin
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.imshow(image)

    # if occluded use dashed else use solid
    linestyle = 'dashed'
    if not box.occluded:
        # width = width * 2
        linestyle = 'solid'

    ax.add_patch(Rectangle(
        (box[0] - width * 0.5, box[1] - width * 0.5), box[2] - box[0], box[3] - box[1], color=color, fill=None, linestyle=linestyle, linewidth=width
    ))

    # convert to rgb
    im = fig2img(fig).convert('RGB')
    plt.close(fig)
    draw = ImageDraw.Draw(im)

    if font:
        ypos = box.ytl
        for attribute in box.attributes:
            attribute = str(attribute).decode("utf-8")
            size = draw.textsize(attribute, font = font)
            xpos = max(box.xtl - size[0] - 3, 0)

            draw.text((xpos, ypos+1), attribute,
                      fill="black", font=font)
            draw.text((xpos+1, ypos+1), attribute,
                      fill="black", font=font)
            draw.text((xpos+1, ypos), attribute,
                      fill="black", font=font)
            draw.text((xpos, ypos-1), attribute,
                      fill="black", font=font)
            draw.text((xpos-1, ypos-1), attribute,
                      fill="black", font=font)
            draw.text((xpos-1, ypos), attribute,
                      fill="black", font=font)

            draw.text((xpos, ypos), attribute,
                      fill="white", font=font)
            ypos += size[1] + 3
    return im

def highlight_boxes(image, boxes, colors = colors, width = defaultwidth,
                    font = None):
    """
    Highlights an iterable of boxes.
    """
    for box, color in zip(boxes, itertools.cycle(colors)):
        highlight_box(image, box, color, width, font)
    return image

def highlight_path(images, path, color = colors[0], width = defaultwidth,
                   font = None):
    """
    Highlights a path across many images. The images must be indexable
    by the frame. Produces a generator.
    """
    logger.info("Visualize path of length {0}".format(len(path)))
    for box in path:
        try:
            lost = box.lost
        except:
            lost = False
        image = images[box.frame]
        if not lost:
            highlight_box(image, box, color, width, font)
        yield image, box.frame

def highlight_paths(images, paths, colors = colors, width = defaultwidth,
                    font = None):
    """
    Highlights multiple paths across many images. The images must be indexable
    by the frame. Produces a generator.
    """

    logger.info("Visualize {0} paths".format(len(paths)))

    boxmap = {}
    paths = zip(paths, itertools.cycle(colors))

    for path, color in paths:
        for box in path:
            if box.frame not in boxmap:
                boxmap[box.frame] = [(box, color)]
            else:
                boxmap[box.frame].append((box, color))

    for frame, boxes in sorted(boxmap.items()):
        im = images[frame]
        for box, color in boxes:
            try:
                lost = box.lost
            except:
                lost = False
            if not lost:
                im = highlight_box(im, box, color, width, font)
        yield im, frame

def save(images, output):
    """
    Saves images produced by the path iterators.
    """
    for image, frame in images:
        image.save(output(frame))
