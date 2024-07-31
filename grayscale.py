'''Imports necessary libraries'''
import os
import time
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
from PIL import Image

def is_grayscale_fast(image_path):
    '''Determines if an image is grayscale'''

    try:
        with Image.open(image_path) as img:
            if img.mode == 'L':
                return image_path

            img_array = np.array(img)
            if img_array.ndim == 2 or img_array.shape[2] == 1:
                return image_path

            # Check a sample of pixels
            sample_size = 1000
            h, w = img_array.shape[:2]
            samples = img_array[np.random.randint(0, h, sample_size),
                                np.random.randint(0, w, sample_size)]

            if np.max(np.std(samples, axis=1)) < 0.1:
                return image_path
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

    return None

def process_chunk(chunk):
    '''Processes a chunk of image file paths to identify grayscale images'''
    return [is_grayscale_fast(img) for img in chunk]

def find_grayscale_images(root_dir, chunk_size=1000):
    '''Find all grayscale images in a directory and its subdirectories'''
    all_images = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith((".jpg")):
                all_images.append(os.path.join(subdir, file))

    print(f"Total images found: {len(all_images)}")

    start_time = time.time()

    # Use all available CPU cores
    num_processes = 12
    print(f"Using {num_processes} processes")

    grayscale_images = []

    with mp.Pool(processes=num_processes) as pool:
        chunks = [all_images[i:i + chunk_size] for i in range(0, len(all_images), chunk_size)]
        for result in tqdm(pool.imap(process_chunk, chunks),
                        total=len(chunks),
                        desc="Processing chunks"):
            grayscale_images.extend([img for img in result if img is not None])

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nTotal images processed: {len(all_images)}")
    print(f"Grayscale images detected: {len(grayscale_images)}")
    print(f"Percentage grayscale: {len(grayscale_images) / len(all_images) * 100:.2f}%")
    print(f"Time taken: {elapsed_time:.2f} seconds ({elapsed_time/3600:.2f} hours)")

    return grayscale_images

def save_grayscale_list(grayscale_images, output_file):
    '''Save thge list of grayscale images '''
    with open(output_file, 'w') as f:
        for img_path in grayscale_images:
            f.write(f"{img_path}\n")
    print(f"\nList of grayscale images saved to: {output_file}")

if __name__ == "__main__":
    root_dir = "img_data"
    output_file = "grayscale_images_to_delete.txt"
    tolerance = 0.1
    chunk_size = 1000

    grayscale_images = find_grayscale_images(root_dir, chunk_size)
    save_grayscale_list(grayscale_images, output_file)

    print("\nTo delete these images, you can use the following command:")
    print(f"xargs -a {output_file} rm")
    print("\nWARNING: Be careful with the deletion. Make sure to review the list before deleting.")