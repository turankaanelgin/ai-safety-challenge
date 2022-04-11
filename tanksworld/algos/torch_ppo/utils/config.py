from .normalizer import *


class Config:

    def __init__(self):

        self.reward_normalizer = RescaleNormalizer()