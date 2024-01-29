from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
def get_main_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg = parser.add_argument
    arg("--weight_path", type=str, default="./Checkpoint_weights/*.h5", help="weights.h5 file")
    arg("--image_path", type=str, default="./Test_images/*.tif", help="tif file")
    arg("--save_path", type=str, default="./prediction.tif", help="output tif file")
    arg("--batch_size", type=int, default=8, help="batch size each predict, lowering to reduce memory requirement")
    return parser.parse_args()

import sys, glob
sys.path.append("./Utility")
sys.path.append("./Model")
from Inference import Window
from Configs.Manual import get_args
from Convolution.Custom import uNet

def predict(weight_path, image_path, save_path, batch_size = 8):
  """
    Args:
        weight_path: weight for the model, .weights.h5 format
        image_path: path to image tif file to predict
        save_path: tif image path to save to
  """
  # take any path that match
  weight_path = glob.glob(f"{weight_path}", recursive=True)[0]
  image_path = glob.glob(f"{image_path}", recursive=True)[0]
  # save_path = save_path
  def get_model(weight_path, args):
    model = uNet.dilatedUNet1(**args, head = "sigmoid")
    model.load_weights(weight_path)
    return model
  
  args = get_args()
  model = get_model(weight_path, args)
  preprocess = lambda x: x/ 255

  Window.predict_windows(pathTif = image_path, pathSave = save_path, 
                  model = model, preprocess = preprocess, batch_size = batch_size,
                  window_size = args["output_size"], predict_dim = args["n_class"])



if __name__ == "__main__":
    main_args = get_main_args()
    predict(**vars(main_args))
