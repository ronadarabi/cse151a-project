import cv2
import os
from pathlib import Path
from tqdm import tqdm

def center_crop(image, crop_size=(512, 512)):
    h, w, _ = image.shape
    ch, cw = crop_size
    start_x = w // 2 - cw // 2
    start_y = h // 2 - ch // 2
    return image[start_y:start_y+ch, start_x:start_x+cw]

def process_images(input_folder, crop_size=(512, 512)):
    input_path = Path(input_folder)

    # Count total files
    total_files = sum(1 for _ in input_path.rglob('*') if _.is_file())

    # Process files with tqdm progress bar
    with tqdm(total=total_files, desc="Processing Images") as pbar:
        for image_file in input_path.rglob('*'):
            if image_file.is_file():
                image = cv2.imread(str(image_file))
                if image is None:
                    print(f"Could not read image: {image_file}")
                    pbar.update(1)
                    continue

                h, w, _ = image.shape
                if h < crop_size[0] or w < crop_size[1]:
                    #print(f"Image too small to crop: {image_file.name}, deleting...")
                    os.remove(image_file)
                else:
                    cropped_image = center_crop(image, crop_size)
                    cv2.imwrite(str(image_file), cropped_image)
                    #print(f"Processed {image_file.name}")
                
                pbar.update(1)

input_folder = 'new_data'
process_images(input_folder)
