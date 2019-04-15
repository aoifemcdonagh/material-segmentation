"""
Script segmenting a single image using a Movidius stick on a Raspberry Pi
Similar to other scripts in this directory, but modified for performance on Movidius stick and Raspberry Pi

"""

import sys
import os
import cv2
from openvino.inference_engine import IENetwork, IEPlugin
import logging as log
import numpy as np
from time import time
import argparse
import minc_plotting as minc_plot

classes_color_map = [
    (150, 150, 150),
    (58, 55, 169),
    (211, 51, 17),
    (157, 80, 44),
    (23, 95, 189),
    (210, 133, 34),
    (76, 226, 202),
    (101, 138, 127),
    (223, 91, 182),
    (80, 128, 113),
    (235, 155, 55),
    (44, 151, 243),
    (159, 80, 170),
    (239, 208, 44),
    (128, 50, 51),
    (82, 141, 193),
    (9, 107, 10),
    (223, 90, 142),
    (50, 248, 83),
    (178, 101, 130),
    (71, 30, 204)
]

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", required=True, type=str)
    parser.add_argument("-i", "--image", help="Path to a single image file", required=True,
                        type=str)
    return parser


if __name__ == "__main__":
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)  # Configure logging
    args = build_argparser().parse_args()

    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # Plugin initialization for Movidius stick
    plugin = IEPlugin(device="MYRIAD")

    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    log.info("Preparing input blobs")
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    net.batch_size = len(args.image)  # Should be 1

    # Read and pre-process input images
    image = cv2.imread(args.image).astype(np.float16)  # Will have to load in range [0-1] for resizing prob maps?
    image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW

    # Reshape input layer for image
    net.reshape({input_blob: (1, image.shape[0], image.shape[1], image.shape[2])})

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = plugin.load(network=net)
    del net

    # Start sync inference
    log.info("Starting inference ")
    t0 = time()
    result = exec_net.infer(inputs={input_blob:image})
    log.info("Average running time of one iteration: {} ms".format((time() - t0) * 1000))

    log.info("processing output blob")

    minc_plot.plot_class_map(result)

    log.info("done")

    output = result[out_blob]
    _, _, out_h, out_w = output.shape
    for batch, data in enumerate(output):
        classes_map = np.zeros(shape=(out_h, out_w, 3), dtype=np.int)
        for i in range(out_h):
            for j in range(out_w):
                if len(data[:, i, j]) == 1:
                    pixel_class = int(data[:, i, j])
                else:
                    pixel_class = np.argmax(data[:, i, j])
                classes_map[i, j, :] = classes_color_map[min(pixel_class, 20)]
        out_img = os.path.join(os.path.dirname(__file__), "out_{}.bmp".format(batch))
        cv2.imwrite(out_img, classes_map)
        log.info("Result image was saved to {}".format(out_img))
    del exec_net
    del plugin
