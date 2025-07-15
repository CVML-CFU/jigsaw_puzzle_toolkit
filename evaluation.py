import pandas as pd
import numpy as np

from PIL import Image

class Evaluation:
    def __init__(self):
        self._img_cache: dict[str, Image.Image] = {}
        pass
