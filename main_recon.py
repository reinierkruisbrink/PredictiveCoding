#!/usr/bin/env python3
import subprocess
import numpy as np
import sys
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.ndimage import interpolation
from tqdm import tqdm, trange
# from PIL import Image
import cv2
from skimage.transform import rescale
from math import ceil

plt.style.use("seaborn-whitegrid")

# ================================ Globals ================================ #
SENSORY_DIM_BASE = 3
SENSORY_DIM = SENSORY_DIM_BASE ** 2  # Sensory Neurons
PREDICTION_DIM = 3 * 3  # Hidden Layer
ACTIVATION = "relu"
INFERENCE_STEPS = 10000
LEARNING_STEPS = 100
LIVE_LOOPS = 1
FACTOR = 3
LEARNING_RATE = 1 * (10 ** -FACTOR)

DIAMOND = "diamond"
CROSS = "cross"
PLUS = "plus"
SQUARE = "square"
POINT = "point"
VLINE = "vline"
HLINE = "hline"
FSLASH = "fslash"
BSLASH = "bslash"

SHAPES = {
    DIAMOND: [[0], [1], [0], [1], [0], [1], [0], [1], [0]],
    CROSS: [[1], [0], [1], [0], [1], [0], [1], [0], [1]],
    PLUS: [[0], [1], [0], [1], [1], [0], [0], [1], [0]],
    SQUARE: [[1], [1], [1], [1], [0], [1], [1], [1], [1]],
    POINT: [[0], [0], [0], [0], [1], [0], [0], [0], [0]],
    VLINE: [[0], [1], [0], [0], [1], [0], [0], [1], [0]],
    HLINE: [[0], [0], [0], [1], [1], [1], [0], [0], [0]],
    FSLASH: [[0], [0], [1], [0], [1], [0], [1], [0], [0]],
    BSLASH: [[0], [0], [1], [0], [1], [0], [1], [0], [0]],
}

SSE_STACK = []
ERR_STACK = []
RECONSTRUCTION_STACK = []
PREDICTION_STACK = []
GEN_STACK = []
SENSE_STACK = []

SAVE_DIR = "recon"
PRINT_OUT_DIM = 500 / SENSORY_DIM_BASE
DURATION = 15  # seconds

# ================================ Functional ================================ #
def save_fig(
    image: np.array,
    width: int,
    height: int,
    name: str = "test",
    scale: int = PRINT_OUT_DIM,
) -> None:
    # image_rescaled = rescale(image.reshape(width, height), scale, anti_aliasing=False)
    # im = Image.fromarray(image.reshape(width, height))
    # im.save(f"./{name}.png")
    cv2.imwrite('filename.jpeg', image.reshape(width, height))
    # plt.imsave(f"./{name}.png", image_rescaled, cmap=plt.cm.jet)


def activation(
    reconstruction: np.array, func: str = "sigmoid", normalize: bool = True
) -> np.array:
    # prediction = weights @ predictions  # r=Wy
    if func == "linear":
        prediction = reconstruction
    elif func == "sigmoid":
        prediction = 1 / (1 + np.exp(-reconstruction))
    elif func == "relu":
        prediction = reconstruction * (reconstruction > 0)
    else:
        try:
            raise UnboundLocalError(f"Illigal 'func' argument: {func}")
        except Exception as err:
            sys.stdout.write(f"Caught other err: {repr(err)}\n")
            raise
    if normalize:
        norm = np.linalg.norm(prediction)
        return prediction / norm
    else:
        return prediction


# ================================ Lower Network ================================ #
def reconstruct(generative_model: np.array, prediction_units: np.array) -> np.array:
    reconstruction = generative_model @ prediction_units
    if reconstruction.shape[0] != SENSORY_DIM:
        try:
            raise ValueError(
                f"Dimension mismatch : {reconstruction.shape} != {SENSORY_DIM}"
            )
        except Exception as error:
            sys.stdout.write(f"Caught other error: {repr(error)}\n")
            raise
    return activation(reconstruction, ACTIVATION)


def SSE(reconstruction: np.array, sensory_input: np.array) -> list:
    """Sum of Squared Errors"""
    residual_error = sensory_input - reconstruction
    return (np.sum(residual_error ** 2), residual_error.reshape((SENSORY_DIM, 1)))


def coefficients(generative_model: np.array, error_units: np.array) -> np.array:
    return np.einsum("ik,ij->k", generative_model, error_units).reshape(
        (PREDICTION_DIM, 1)
    )


# ================================ High Network ================================ #
def inference(
    generative_model: np.array, prediction_units: np.array, sensory_input: np.array
) -> list:
    global SSE_STACK, ERR_STACK, RECONSTRUCTION_STACK, PREDICTION_STACK

    for __ in range(INFERENCE_STEPS):
        reconstruction = reconstruct(generative_model, prediction_units)
        RECONSTRUCTION_STACK.append(reconstruction)

        error, error_units = SSE(reconstruction, sensory_input)
        SSE_STACK.append(error)
        ERR_STACK.append(error_units)

        prediction_units = coefficients(generative_model, error_units)
        PREDICTION_STACK.append(prediction_units)
    return (prediction_units, sensory_input)


def learning(
    generative_model: np.array, error_units: np.array, prediction_units: np.array
) -> None:
    old_generative_model = deepcopy(generative_model)
    for __ in enumerate(range(LEARNING_STEPS)):
        gen_model_delta = np.einsum("ik,jk->ji", error_units, prediction_units).reshape(
            (SENSORY_DIM, PREDICTION_DIM)
        )
        generative_model += gen_model_delta * LEARNING_RATE
        GEN_STACK.append(generative_model)
        if np.array_equal(old_generative_model, generative_model):
            break
        old_generative_model = deepcopy(generative_model)
    return activation(generative_model, ACTIVATION)


# ================================ Bash Commands Setup ================================ #
def send_command(command: str) -> None:
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

def make_video(fps: int) -> None:
    sys.stdout.write(f"\n # --- RUNNING FFMPEG COMMAND: FPS {fps} --- # \n")
    ffmpeg = f"ffmpeg -r {fps} -f image2 -s 1920x1080 -i ./recon/%04d.png  -vcodec libx264 -crf 25  -pix_fmt yuv420p reconstruction.mp4 -y"
    send_command(ffmpeg)

# ================================ Graphics ================================ #
def info_graphics() -> None:
    sys.stdout.write("\n # === SAVING RECONSTRUCTION RESULTS LOCALLY === # \n")

    # Error
    fig = plt.figure()
    ax = plt.axes()
    x = np.linspace(0, 10, len(SSE_STACK))
    ax.plot(x, SSE_STACK)
    plt.savefig("error")

    # Remove Recon Folder (Delete Old Data)
    rmv_fldr = "rm -rf ./recon"
    send_command(rmv_fldr)

    # Make Recon Folder (Clean Destination Folder For Images)
    mk_fldr = "mkdir ./recon"
    send_command(mk_fldr)

    # Video
    printable = RECONSTRUCTION_STACK

    save_fig(SENSE_STACK[0], SENSORY_DIM_BASE, SENSORY_DIM_BASE, "input")
    for idx, recon in tqdm(enumerate(printable), total=len(printable)):
        save_fig(recon, SENSORY_DIM_BASE, SENSORY_DIM_BASE, f"{SAVE_DIR}/{idx:04d}")

    fps = ceil(len(printable) / DURATION)
    # make_video(fps=fps)

# ================================ Graphics ================================ #
def main() -> None:
    global SSE_STACK, ERR_STACK, RECONSTRUCTION_STACK, PREDICTION_STACK, SENSE_STACK

    # sensory_input = np.random.uniform(
    #     low=0.0, high=1.0, size=(SENSORY_DIM, 1)
    # )  #  bottom-most layer / signals delivered by external sensory organs
    sensory_input = np.array(SHAPES[CROSS])
    SENSE_STACK.append(sensory_input)

    prediction_units = np.random.normal(  # -> Change on inference
        loc=0.5, scale=0.1, size=(PREDICTION_DIM, 1)
    )  # Prediction layer / coefficients
    PREDICTION_STACK.append(prediction_units)

    error_units = np.random.normal(
        loc=0.5, scale=0.1, size=(SENSORY_DIM, 1)
    )  # Prediction layer / coefficients
    ERR_STACK.append(error_units)

    generative_model = np.random.normal(  # -> Change on learning
        loc=0.01, scale=0.005, size=(SENSORY_DIM, PREDICTION_DIM)
    )  # Weights
    GEN_STACK.append(generative_model)

    sys.stdout.write("\n # === STARTING SIMULATION (LIVE LOOPS) === # \n")
    for __ in trange(LIVE_LOOPS):
        # Inference
        prediction_units, sensory_input = inference(
            generative_model, prediction_units, sensory_input
        )

        # Learning
        generative_model = learning(generative_model, prediction_units, sensory_input)

    info_graphics()

if __name__ == "__main__":
    main()
