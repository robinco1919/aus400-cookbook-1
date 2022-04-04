import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import io
import PIL
from  .render import field_to_image, set_image_mpl_cmap

# get the lat and long of the centre of a DataArray
# warning: may not be a real point in dataset (use find_nearest_index to correct)
def get_centre(data):
    lat = (data.latitude.max() + data.latitude.min())/2
    long = (data.longitude.max() + data.longitude.min())/2
    return lat.item(), long.item()

# save a path (list of lat, long points) to a text file so it doesn't have to be recalculated
def save_path(path, name):
    file = open(f'Aus400/saved_paths/{name}.txt', 'w')
    for item in path:
        file.write("{} {}\n".format(item[0], item[1]))
    file.close()

# load a previously saved path
def load_path(name, is_int = False):
    path = []
    file = open(f'Aus400/saved_paths/{name}.txt', 'r')
    for line in file:
        split_line = line.strip().split()
        if (is_int):
            path.append((int(split_line[0]), int(split_line[1])))
        else:
            path.append((float(split_line[0]), float(split_line[1])))
    file.close()
    return path

def get_preset_size(preset):
    if preset == "360p":
        return 640, 360
    elif preset == "480p":
        return 854, 480
    elif preset == "720p":
        return 1280, 720
    elif preset == "1080p":
        return 1920, 1080
    elif preset == "4K":
        return 3840, 2160
    elif preset == "8K":
        return 7680, 4320
    else:
        raise ValueError("Unknown preset - try set_size instead")
    

# smooth a line by taking a moving average (size according to tolerance parameter) for every point (except end points)
def smooth_line(points, tolerance = 1):
    smooth_points = [points[0]]
    total_points = len(points)
    for i in range(1, total_points - 1):
        adj_tol = min(tolerance, i, total_points - i - 1)
        lat_avg = sum([x[0] for x in points[i-adj_tol:i+adj_tol+1]]) / (adj_tol * 2 + 1)
        long_avg = sum([x[1] for x in points[i-adj_tol:i+adj_tol+1]]) / (adj_tol * 2 + 1)
        smooth_points.append((lat_avg, long_avg))
    smooth_points.append(points[-1])
    return smooth_points

# extend a line by adding filler points between existing points
def interpolate_line(points, multiplier):
    path = []
    for i in range(len(points) - 1):
        lat_diff = (points[i + 1][0] - points[i][0]) / multiplier
        long_diff = (points[i + 1][1] - points[i][1]) / multiplier
        for j in range(multiplier):
            path.append((points[i][0] + j * lat_diff, points[i][1] + j * long_diff))
    path.append(points[-1])
    return path
    

# find the lat, long point in a DataArray closest to a different given point
def find_nearest_index(data, point):
    nearest_point = data.sel(latitude = point[0], longitude = point[1], method = "nearest")
    lat_pos = np.where(data.latitude == nearest_point.latitude)[0][0]
    long_pos = np.where(data.longitude == nearest_point.longitude)[0][0]
    return lat_pos, long_pos


def snap_line(data, points):
    return [find_nearest_index(data, point) for point in points]

# source: https://stackoverflow.com/a/61755066
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img

def create_colorbar(label, cmap, height, vmin, vmax, reduction = 2):
    
    fig, ax = plt.subplots(figsize=(reduction, reduction * 4), dpi = height // (reduction * 4))
    fig.subplots_adjust(right=0.25)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=ax, orientation='vertical', label=label)
    
    plt.close()
    
    return fig2img(fig)

def simple_downsample(data, template):
    return data.sel(latitude = template.latitude.values, longitude = template.longitude.values, method = 'nearest')

def create_border(data, thickness):
    new_data = data - data
    for i in range(-thickness, thickness + 1):
        for j in range(-thickness, thickness + 1):

            # skip no shift case
            if i == 0 and j == 0:
                continue

            data_shift = data.shift(latitude = i, longitude = j, fill_value = 0)
            new_data = new_data + data.where(data != data_shift, other = 0)   
    return new_data.where(new_data == 0, other = 1)

def create_border_image(data, thickness):
    return field_to_image(create_border(data, thickness), vmax = 1)

def crop_image_index(image, lat_min_pos, lat_max_pos, long_min_pos, long_max_pos):
    return image.crop((long_min_pos, image.height - lat_max_pos, long_max_pos + 1, image.height - lat_min_pos + 1))

def create_data_image(data, cmap, border_mask, vmin, vmax):
    image = field_to_image(data, vmin = vmin, vmax = vmax)
    if image.size != border_mask.size:
        raise ValueError(
            "Data size and border mask size do not match - "
            + "try using crop_image_index to match the border")
    c_image = set_image_mpl_cmap(image, cmap)
    c_image.paste(0, mask = border_mask)
    return c_image.convert('RGBA')
    