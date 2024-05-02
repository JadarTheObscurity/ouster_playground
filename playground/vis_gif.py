# given the folder with images, create a gif in the same folder

import os
from PIL import Image
import shutil
import subprocess
def create_gif(folder_path, gif_name):
    images = []
    # Check if file not found error
    if not os.path.exists(folder_path):
        print(f"[!] Folder not found: {folder_path}")
        return
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            images.append(image)

    gif_path = os.path.join(folder_path, gif_name)
    images[0].save(gif_path, save_all=True, append_images=images[1:], optimize=False, duration=200, loop=0)

# Example usage
folder_path = r"."
gif_name = "animation.gif"
create_gif(folder_path, gif_name)