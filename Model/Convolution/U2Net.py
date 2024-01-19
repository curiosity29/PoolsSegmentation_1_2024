import tensorflow as tf
from tensorflow.keras import layers
from Blocks.Basic import CoBaRe

class RSU(layers.Layer):
  def __init__(self, height, in_ch, mid_ch, out_ch, dilation = 1, **kwargs):

    super(RSU, self).__init__(**kwargs)
    args = (height, in_ch, mid_ch, out_ch, dilation)
    self.height = height

    self.model = self.get_model(*args)

  @staticmethod
  def get_model(height, in_ch, mid_ch, out_ch, dilated):
    def cbr(x, name, filters = mid_ch):
      return CoBaRe(filters = filters)(x)

    input = tf.keras.Input(shape = (None, None, in_ch), name = "RSU_input")
    x = cbr(input, filters = out_ch)
    skips = [x]
    x = cbr(x, name = "rs_en_1")
    skips.append(x)
    ## down
    for idx in range(2, height):
      x = layers.MaxPool2D(2, name = f"down_{idx}")(x)
      x = cbr(x, name = f"rs_en_{idx}")
      skips.append(x)
      # print(x.shape)

    # middle
    x = cbr(x, name = f"rs_en_{height}")

    x = layers.Concatenate(axis = -1, name = f"skip_{height-1}")([x, skips.pop()])
    x = cbr(x, name = f"rs_de_{height-1}")
    # print("decoding")
    for idx in range(height-2, 1, -1):
      x = layers.UpSampling2D(2, name = f"up_{idx}")(x)
      # print(x.shape)
      x = layers.Concatenate(axis = -1, name = f"skip_{idx}")([x, skips.pop()])
      x = cbr(x, name = f"rs_de_{idx}")

    x = layers.UpSampling2D(2, name = f"up_1")(x)
    x = layers.Concatenate(axis = -1, name = "skip_1")([x, skips.pop()])
    x = cbr(x, name = "rs_de_1", filters = out_ch)
    output = layers.Add(name = "add_ouput")([x, skips.pop()])

    return tf.keras.Model(input, output)

##### NEED CONFIG MODEL

  def get_config(self):
    config = super().get_config()
    config.update(
      {
        "conv": self.conv,
        "batch": self.batch
      }
    )
    return config

  @classmethod
  def from_config(cls, config):
    config["conv"] = tf.keras.layers.deserialize(config["conv"])
    config["batch"] = tf.keras.layers.deserialize(config["batch"])
        
    return cls(**config)

##### NEED CONFIG MODEL


  def call(self, inputs):
    # Apply the layer's logic here
    return self.model(inputs)


configs1 = {
    # cfgs for building RSUs and sides
    # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilation), side]}
    'stage1': ['En_1', (7, 3, 32, 64, 1), -1],
    'stage2': ['En_2', (6, 64, 32, 128, 1), -1],
    'stage3': ['En_3', (5, 128, 64, 256, 1), -1],
    'stage4': ['En_4', (4, 256, 128, 512, 1), -1],
    'stage5': ['En_5', (4, 512, 256, 512, 1), -1],
    'stage6': ['En_6', (4, 512, 256, 512, 1), 512],
    'stage5d': ['De_5', (4, 1024, 256, 512, 1), 512],
    'stage4d': ['De_4', (4, 1024, 128, 256, 1), 256],
    'stage3d': ['De_3', (5, 512, 64, 128, 1), 128],
    'stage2d': ['De_2', (6, 256, 32, 64, 1), 64],
    'stage1d': ['De_1', (7, 128, 16, 64, 1), 64],
}

# def side(x, target_size):
def get_side(x, filters, target_size):
  x = layers.Conv2D(filters = filter, kernel_size = 1)(x)
  x = layers.Activation("sigmoid")(x)
  x = layers.UpSampling2D(target_size // x.shape[1])(x)
  return x

def U2Net(n_channel = 3, n_class = 2, input_size = 256, output_size = 256, 
  softmax_head = True, configs = configs1, **ignore):
  n_en = (len(configs) - 3) // 2
  configList = list(configs.values())
  configs_en, configs_middle, configs_de = \
  configList[:n_en], configList[n_en: n_en + 3], configList[-n_en:]
  inputs = tf.keras.Input(shape = (input_size, input_size, n_channel))
  x = inputs
  skips = []
  sides = []
  for idx, config_ in enumerate(configs_en):
    name, (height, in_ch, mid_ch, out_ch, dilation), side = config_
    x = RSU(height, in_ch, mid_ch, out_ch, dilation, name = name)
    x = layers.MaxPool2D(2, name = f"down_{idx+1}")(x)
    skips.append(x)

  ### bottle neck
  for config_ in configs_middle:
    name, (height, in_ch, mid_ch, out_ch, dilation), side = config_
    x = RSU(height, in_ch, mid_ch, out_ch, dilation, name = name)
    if side > 0:
      sides.append(get_side(x, filters = side, target_size = output_size))

  for idx, config_ in enumerate(configs_de):
    name, (height, in_ch, mid_ch, out_ch, dilation), side = config_
    x = RSU(height, in_ch, mid_ch, out_ch, dilation, name = name)
    x = layers.Concatenate(axis = -1, name = "skip_{n_en - idx}")
    x = layers.UpSampling2D(2, name = f"up_{n_en - idx}")(x)
    skips.append(x)
  

  ### segment head





