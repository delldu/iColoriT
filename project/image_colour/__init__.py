"""Image Color Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2022(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2022年 09月 04日 星期日 16:41:59 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
import time
from tqdm import tqdm
import torch

import redos
import todos

from . import data, color

import pdb


def get_model():
    """Create model."""

    model_path = "models/image_colour.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = color.Colour()

    todos.model.load(model, checkpoint)
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    model = torch.jit.script(model)
    todos.data.mkdir("output")
    if not os.path.exists("output/image_colour.torch"):
        model.save("output/image_colour.torch")

    return model, device

def model_forward(model, device, input_tensor, multi_times=1):
    # zeropad for model
    H, W = input_tensor.size(2), input_tensor.size(3)
    if H % multi_times != 0 or W % multi_times != 0:
        input_tensor = todos.data.zeropad_tensor(input_tensor, times=multi_times)

    output_tensor = todos.model.forward(model, device, input_tensor)

    return output_tensor[:, :, 0:H, 0:W]


def image_client(name, input_files, output_dir):
    redo = redos.Redos(name)
    cmd = redos.image.Command()
    image_filenames = todos.data.load_files(input_files)
    for filename in image_filenames:
        output_file = f"{output_dir}/{os.path.basename(filename)}"
        context = cmd.colour(filename, output_file)
        redo.set_queue_task(context)
    print(f"Created {len(image_filenames)} tasks for {name}.")


def image_server(name, host="localhost", port=6379):
    # load model
    model, device = get_model()

    def do_service(input_file, output_file, targ):
        print(f"  clean {input_file} ...")
        try:
            input_tensor = todos.data.load_rgba_tensor(input_file)
            output_tensor = model_forward(model, device, input_tensor)
            todos.data.save_tensor(output_tensor, output_file)
            return True
        except Exception as e:
            print("exception: ", e)
            return False

    return redos.image.service(name, "image_colour", do_service, host, port)


def image_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_rgba_tensor(filename)
        input_tensor = data.color_sample(input_tensor, 0.02)

        # pytorch recommand clone.detach instead of torch.Tensor(input_tensor)
        orig_tensor = input_tensor.clone().detach()
        predict_tensor = model_forward(model, device, input_tensor)
        output_file = f"{output_dir}/{os.path.basename(filename)}"

        todos.data.save_tensor([orig_tensor[:, 0:3, :, :], predict_tensor], output_file)
