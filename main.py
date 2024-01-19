from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
def get_main_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg = parser.add_argument
    arg("--weight_path", type=str, default="./checkpoint.weights.h5", help="")
    arg("--image_path", type=str, default="./image.tif", help="")
    arg("--save_path", type=str, default="./predict.tif", help="")
    return parser.parse_args()

from Utility import Inference, Configs
from Model import Blocks, Convolution
from Utility.Inference import Window
from Utility.Configs.Manual import get_args
from Model.Convolution.Custom import uNet

def predict(weight_path, image_path, save_path):
  """
    Args:
        weight_path: weight for the model, .weights.h5 format
        image_path: path to image tif file to predict
        save_path: tif image path to save to
  """
  def get_model(weight_path, args):
    model = uNet.dilatedUNet1(**args, head = "sigmoid")
    model.load_weights(weight_path)
    return model
  
  args = get_args()
  model = get_model(weight_path, args)
  preprocess = lambda x: x/ 255

  Window.predict_windows(pathTif = image_path, pathSave = save_path, 
                  model = model, preprocess = preprocess,
                  widow_size = args["output_size"], predict_dim = args["n_class"])



if __name__ == "__main__":
    main_args = get_main_args()
    predict(**main_args)
