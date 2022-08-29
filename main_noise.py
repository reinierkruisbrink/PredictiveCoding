#!/usr/bin/env python3

"""
STUDY Inverses of shapes
STUDY Scales, activations, & initalization values
STUDY MMN as a consequence of inference or of learning

ERROR Large LIVE_LOOPS number breaks code (linear)

[ ] Inhibotory and Excitatory diff (80/20)
[ ] More layers
[ ] Precision weighting
[ ] Translation / Rotation
[ ] Larger window


[x] Noise modelling / Peturbation
[x] Presenting new input
"""

import numpy as np
import sys
from copy import deepcopy
import matplotlib.pyplot as plt
from random import randint
from tqdm import tqdm

plt.style.use("seaborn-whitegrid")

SENSE_LAYER_DIM, HIDDEN_LAYER_DIM = 9, 9

INFERENCE_STEPS = 1000
LEARNING_STEPS = 100
FACTOR = 3
LEARNING_RATE = 1 * (10 ** -FACTOR)
LIVE_LOOPS = 20
ACTIVATION = "sigmoid"
BASAL_FACTOR = 100

DIAMOND = "diamond"
CROSS = "cross"
PLUS = "plus"
SQUARE = "square"
POINT = "point"
VLINE = "vline"
HLINE = "hline"
FSLASH = "fslash"
BSLASH = "bslash"
EMPTY = "empty"

SHAPES = {
    "diamond": [[0], [1], [0], [1], [0], [1], [0], [1], [0]],
    "cross": [[1], [0], [1], [0], [1], [0], [1], [0], [1]],
    "plus": [[0], [1], [0], [1], [1], [0], [0], [1], [0]],
    "square": [[1], [1], [1], [1], [0], [1], [1], [1], [1]],
    "point": [[0], [0], [0], [0], [1], [0], [0], [0], [0]],
    "vline": [[0], [1], [0], [0], [1], [0], [0], [1], [0]],
    "hline": [[0], [0], [0], [1], [1], [1], [0], [0], [0]],
    "fslash": [[0], [0], [1], [0], [1], [0], [1], [0], [0]],
    "bslash": [[0], [0], [1], [0], [1], [0], [1], [0], [0]],
    "empty": [[0], [0], [0], [0], [0], [0], [0], [0], [0]],
}


def model(weights: np.array, predictions: np.array, func: str = "linear") -> np.array:
    try:
        prediction = np.einsum("ki,kj->ij", weights, predictions)  # r=Wy
    except ValueError as error:
        sys.stdout.write(f"Caught Other Error: {error}\n")
        raise

    #prediction = weights @ predictions  # r=Wy
    if func == "linear":
        return prediction
    if func == "sigmoid":
        return 1 / (1 + np.exp(-prediction))
    elif func == "relu":
        return prediction * (prediction > 0)
    else:
        try:
            raise UnboundLocalError(f'Illigal "func" arguement: {func}')
        except Exception as err:
            sys.stdout.write(f"Caught other err: {repr(err)}\n")
            raise


def err(target, prediction, func: str = "sse") -> list:
    error_units = target - prediction
    if func == "sse":
        return (error_units, np.sum(error_units ** 2))
    elif func == "mse":
        return (error_units, np.mean(error_units ** 2))
    else:
        try:
            raise UnboundLocalError(f'Illigal "func" arguement: {func}')
        except Exception as err:
            sys.stdout.write(f"Caught other err: {repr(err)}\n")
            raise


class PC_layer:
    """Single layer of predictive coding"""

    def __init__(self, sensory_dim: int, prediction_dim: int) -> None:
        # Stats
        self.total_err_log = []
        self.sense_stack = []

        self.prediction_units = np.random.normal(
            loc=0.5, scale=0.1, size=(prediction_dim, 1)
        )  # Prediction layer / coefficients

        self.error_units = np.random.normal(
            loc=0.5, scale=0.1, size=(sensory_dim, 1)
        )  # Prediction layer / coefficients

        self.generative_model = np.random.normal(
            loc=0.01, scale=0.005, size=(sensory_dim, prediction_dim)
        )  # Weights

    def basal_noise(self, shape: tuple = (1, 1)) -> float:
        basal_noise = np.random.uniform(
            low=-1, high=1, size=shape
        )  # Basal Firing Rate (0.0667 -> 0.133)
        return basal_noise / BASAL_FACTOR

    @property
    def sensory_input(self):
        return self._sensory_input

    @sensory_input.setter
    def sensory_input(self, sense_in: np.array):
        # TODO Do checks for connectivity ?
        if callable(sense_in):
            noise = sense_in(shape=self._sensory_input.shape)
            self._sensory_input = self._sensory_input + (noise)
        else:
            self._sensory_input = sense_in
            self.sense_stack.append(deepcopy(self._sensory_input))

    def compute_prediction(self) -> None:
        self.reconstruction = model(
            self.generative_model, self.error_units, ACTIVATION
        )

    def compute_sensory_prediction_err(self) -> None:
        self.error_units, self.total_err = err(
            self.sensory_input, self.reconstruction, "sse"
        )
        self.total_err_log.append(deepcopy(self.total_err))

    def quanta(self) -> None:
        self.generative_model += self.basal_noise(shape=self.generative_model.shape)
        # self.prediction_units += self.basal_noise(shape=self.prediction_units.shape)
        # self.error_units += self.basal_noise(shape=self.error_units.shape)
        # self.sensory_input = self.basal_noise
        self.compute_prediction()
        self.compute_sensory_prediction_err()

    def inference(self) -> None:
        for __ in range(INFERENCE_STEPS):
            self.quanta()
            new_prediction = model(self.generative_model, self.prediction_units, ACTIVATION)
            if np.sum(new_prediction - self.prediction_units) == 0.0:
                break
            self.prediction_units = new_prediction

    def learning(self) -> None:
        for __ in range(LEARNING_STEPS):
            self.quanta()
            gen_model_delta = np.einsum(
                "ik,ji->ij", self.error_units, self.prediction_units
            )
            self.generative_model += gen_model_delta * LEARNING_RATE

    def infographics(self) -> None:
        fig = plt.figure(figsize=(10, 8))
        plt.title(
            f"ACTI: {ACTIVATION} INF: {INFERENCE_STEPS} LR: {LEARNING_STEPS} LRRF:{FACTOR} LL: {LIVE_LOOPS} BF: {BASAL_FACTOR}"
        )
        grid = plt.GridSpec(2, len(self.sense_stack) + 2, wspace=0.4, hspace=0.3)

        error_log = fig.add_subplot(grid[0, :])
        x = np.linspace(0, 1, len(self.total_err_log))
        error_log.plot(x, self.total_err_log)

        for i, sense in enumerate(self.sense_stack):
            target = fig.add_subplot(grid[1, i])
            img_dim = int(np.sqrt(sense.shape[0]))
            x = sense.reshape(img_dim, img_dim)
            print(x)
            target.imshow(x)

        target = fig.add_subplot(grid[1, i + 1])
        img_dim = int(np.sqrt(self.reconstruction.shape[0]))
        x = self.reconstruction.reshape(img_dim, img_dim)
        print(x)
        target.imshow(x)

        target = fig.add_subplot(grid[1, i + 2])
        img_dim = int(np.sqrt(self.sensory_input.shape[0]))
        x = self.sensory_input.reshape(img_dim, img_dim)
        print(x)
        target.imshow(x)

        fig.savefig(
            f"./{ACTIVATION}_{INFERENCE_STEPS}_{LEARNING_STEPS}_{FACTOR}_{LIVE_LOOPS}_{BASAL_FACTOR}-{randint(1, 500)}.png"
        )
        # plt.show()

    def handler(
        self,
        shape: str = "diamond",
        infer: bool = True,
        learn: bool = True,
        loops: int = LIVE_LOOPS,
    ) -> None:
        sensory_input = np.array(SHAPES[shape])  # Sensory layer
        self.sensory_input = sensory_input

        for __ in tqdm(range(loops)):
            if infer:
                self.inference()
            if learn:
                self.learning()


if __name__ == "__main__":
    pc = PC_layer(SENSE_LAYER_DIM, HIDDEN_LAYER_DIM)  # Create NN
    pc.handler(DIAMOND, loops=25)  # World dynamics
    pc.handler(SQUARE, loops=25, learn=False)  # World dynamics
    pc.handler(DIAMOND, loops=100, learn=False)  # World dynamics
    pc.infographics()  # Information