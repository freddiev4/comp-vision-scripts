# vision-scripts

Contains Computer Vision scripts I've put together for re-use / quick tasks.

## object detection

```python
python object-detection/detr.py --help
Usage: detr.py [OPTIONS]

  CLI program that pulls a pretrained DETR model from huggingface, and runs it
  on an image passed via a URL. The results are then saved to a file.

  Default model: facebook/detr-resnet-50

  Default filename: detr-bounding-boxes.png

Options:
  --image-url TEXT         URL that links to an image file  [required]
  --pretrained-model TEXT  Name of the pretrained model on huggingface
  --output-file TEXT       Name of the output file that will be saved
  --help                   Show this message and exit.
```