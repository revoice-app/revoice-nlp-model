from predictor import EnsembleBuilder
from train_driver import TrainDriver
import sys


recogspeech = sys.argv[1]
eb = EnsembleBuilder()
result = eb.make_prediction(recogspeech)
print("User Review: {}".format(recogspeech))
print(result)
