import cv2
import numpy as np
from pathlib import Path
import multiprocessing as mp
import gc
from zipfile import ZipFile

## Extract dataset archive
with ZipFile('archive.zip', 'r') as zip:
    zip.extractall('archive/')

# Read image, convert to RGB and Lab colorspaces, and calculate
# frequencies of pixel brightness for each channel in both colorspaces.
# Also find the shape of the image.
def process_image(image_file):
    im_BGR = cv2.imread(str(image_file))
    if im_BGR is None:
        print("Couldn't read image from", image_file)
        return None, None

    im_RGB = cv2.cvtColor(im_BGR, cv2.COLOR_BGR2RGB)
    im_Lab = cv2.cvtColor(im_BGR, cv2.COLOR_BGR2Lab)

    im = np.append(im_RGB, im_Lab, axis=2)
    height, width, num_channels = im.shape

    sizes = np.array([height, width])

    counts_per_channel = np.zeros((num_channels, 256), dtype=np.int64)
    for channel in range(num_channels):
        channel_values = im[:, :, channel]
        counts = np.bincount(channel_values.flatten(), minlength=256)
        counts_per_channel[channel, :] = counts

    return counts_per_channel, sizes

# Add pixel brightness frequencies to global running sum,
# and add image shape to size array
def update_totals(result):
    global counts_per_channel_tot, sizes, idx, failed_reads
    counts_per_channel, size = result
    idx += 1
    if counts_per_channel is not None:
        counts_per_channel_tot += counts_per_channel
        sizes[idx, :] = size
        if idx % 10000 == 0:
            gc.collect()
            print('Processed Image:', idx, flush=True)
    else:
        failed_reads += 1

if __name__ == '__main__':
    # get number of images
    input_dir = 'archive'
    input_path = Path(input_dir)
    num_images = sum(1 for _ in input_path.rglob('*') if _.is_file())

    # initialize arrays, variables
    counts_per_channel_tot = np.zeros((6, 256), dtype=np.int64)
    sizes = np.zeros((num_images, 2), dtype=int)
    idx = 0
    failed_reads = 0

    # loop through images
    pool = mp.Pool(mp.cpu_count() // 2)
    results = [pool.apply_async(process_image, args=(image_file,), callback=update_totals) for image_file in input_path.rglob('*') if image_file.is_file()]

    pool.close()
    pool.join()

    # write arrays to disk
    np.savetxt('RGB_frequencies.csv', counts_per_channel_tot[0:3], delimiter=',')
    np.savetxt('Lab_frequencies.csv', counts_per_channel_tot[3:6], delimiter=',')
    np.savetxt('image_sizes.csv', sizes, delimiter=',')

    print('Sucessfully processed {} images, but failed to load {} images'.format(idx - failed_reads, failed_reads))