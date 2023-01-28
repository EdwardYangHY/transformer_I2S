from PIL import Image,ImageDraw
from glob import glob
import os
from tqdm import tqdm

def image_edit(img_path,color):
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    if color == "b":
        fill = (0,0,255)
    elif color == "r":
        fill = (255,0,0)
    else:
        fill = (0,255,0)
    draw.rectangle(
        [(0, 0), (70, 70)], 
        fill=fill, 
        # outline=(0, 255, 0), 
        width=10
    )
    file_name = img_path.split("/")[-1].split(".")[0]+f"_{color}.jpg"
    img.save(root_path+file_name)

#写真のパスを獲得
filepath = glob("../../data/I2U/image/*/*/*")
#一つごとの編集
for img_path in tqdm(filepath):
    #../../data/I2U/image/apple/test_number1/group1_1.jpg
    #../../data/I2U/image_color/apple/test_number1/group1_1_b.jpg
    root_path = "/".join(img_path.split("/")[:5])+"_color/"+"/".join(img_path.split("/")[5:7])+"/"
    os.makedirs(root_path,exist_ok=True)
    for color in ["r","g","b"]:
        image_edit(img_path,color)
