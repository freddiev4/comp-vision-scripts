import logging
import random

import click
import requests
import torch

from matplotlib import pyplot as plt
from PIL import Image
from transformers import (
    DetrFeatureExtractor,
    DetrForObjectDetection,
    Pipeline,
    pipeline,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DETR test script")


def run_object_detection_pipeline(pipe: Pipeline, image: Image) -> None:
    # set figure so we can draw bounding boxes
    plt.figure(figsize=(16, 10))
    plt.imshow(image)

    # keep track of the set of items, so we can use a new color
    # for each kind of item when we draw the bounding boxes
    items_to_colors = {}
    colors = ["r", "g", "b", "c", "m", "y"]

    for results in pipe(image):
        bbox = results["box"]
        item = results["label"]
        score = round(results["score"], 3)

        # if we run out of colors, we have too many types of objects, so we'll
        # just stop the process.
        # TODO: there's a better way to do this without running out of colors.
        if not colors:
            logger.error("Ran out of available base colors...exiting")
            raise Exception(
                "Too many objects detected... not enough colors to draw boxes"
            )

        # assign a new color if we don't already have the item in our set
        if item not in items_to_colors:
            color = random.choice(colors)
            items_to_colors[item] = color
            colors.remove(color)
        else:
            color = items_to_colors[item]

        # let's only keep detections with score > 0.9
        if score > 0.9:
            logging.info(f"Detected {item} with confidence {score} at location {bbox}")

            logging.info("Drawing bounding boxes...")
            draw_bounding_boxes(
                bbox=bbox,
                item=item,
                score=score,
                color=color,
            )

    return None


def draw_bounding_boxes(
    bbox: dict,
    item: str,
    score: float,
    color: str,
) -> None:
    ax = plt.gca()

    (xmin, ymin, xmax, ymax) = (
        bbox["xmin"],
        bbox["ymin"],
        bbox["xmax"],
        bbox["ymax"],
    )

    rectangle = plt.Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        fill=False,
        color=color,
        linewidth=3,
    )

    rx, ry = rectangle.get_xy()
    ax.add_patch(rectangle)

    ax.annotate(
        text=f"{item}, CF: {score}",
        xy=(rx, ry),
        color="khaki",
        weight="bold",
        fontsize=10,
        ha="left",
        va="bottom",
    )
    return None


@click.command()
@click.option(
    "--image-url",
    help="URL that links to an image file",
    required=True,
)
@click.option(
    "--pretrained-model",
    help="Name of the pretrained model on huggingface",
    default="facebook/detr-resnet-50",
)
@click.option(
    "--output-file",
    help="Name of the output file that will be saved",
    default="detr-bounding-boxes.png",
)
def main(image_url: str, pretrained_model: str, output_file: str):
    """
    CLI program that pulls a pretrained DETR model from huggingface, and runs it
    on an image passed via a URL. The results are then saved to a file.

    Default model: facebook/detr-resnet-50

    Default filename: detr-bounding-boxes.png
    """
    logger.info(f"Getting image at url {image_url}...")
    image = Image.open(requests.get(image_url, stream=True).raw)
    logger.info(f"Now have image object {image}")

    logger.info(f"Getting {pretrained_model} pretrained model...")
    model = DetrForObjectDetection.from_pretrained(pretrained_model)
    feature_extractor = DetrFeatureExtractor.from_pretrained(pretrained_model)

    logger.info("Setting up object detection pipeline...")
    pipe = pipeline(
        "object-detection", model=model, feature_extractor=feature_extractor
    )

    logger.info("Running object detection pipeline...")
    run_object_detection_pipeline(pipe=pipe, image=image)

    plt.savefig(output_file)
    logging.info(f"Saved image to location: {output_file}")


if __name__ == "__main__":
    main()
