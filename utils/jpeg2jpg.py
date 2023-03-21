import glob
import os

def rename_jpeg_to_jpg(root_directory):
    Subdirectorys = glob.glob(root_directory)
    for Subdirectory in Subdirectorys[:]:
        #print(Subdirectory)
        img_path = glob.glob(Subdirectory + "/*.JPEG")
        for img in img_path[:]:
            print(img)
            new_file=img.split(".")[0]+".jpg"
            print(new_file)
            os.rename(img, new_file)

def rename_png_to_jpg(root_directory):
    Subdirectorys = glob.glob(root_directory)
    for Subdirectory in Subdirectorys[:]:
        #print(Subdirectory)
        img_path = glob.glob(Subdirectory + "/*.png")
        for img in img_path[:]:
            print(img)
            new_file=img.split(".")[0]+".jpg"
            print(new_file)
            os.rename(img, new_file)

root_directory = "/home/licheng/hjx/data/view/view1_pad/"
# rename_jpeg_to_jpg(root_directory)
rename_png_to_jpg(root_directory)