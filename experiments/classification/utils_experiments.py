from PIL import Image
from io import BytesIO
import os
from paths import *

# data verification 
classes = os.listdir(TRAIN_DIR)

for _class in classes:
    dir_class = os.path.join(TRAIN_DIR,_class)
    images = os.listdir(dir_class)
    try:
        images.remove(".ipynb_checkpoints")
    except:
        pass
    for _image in images:
        with open(os.path.join(dir_class,_image), "rb") as f:
            image_data = f.read()
        try:
            img = Image.open(BytesIO(image_data))
        except OSError as e:
            print("Error: ", e, " image: ",os.path.join(dir_class,_image))
