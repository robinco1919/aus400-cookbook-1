import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import io
import PIL
from PIL import ImageDraw, ImageFont
from  .render import field_to_image, set_image_mpl_cmap
from  .support import find_nearest_index, fig2img, load_path, get_preset_size
from  .regrid import *


# def regrid_vector(data, grid):
#     """
#     Redrigs vector quantities like u/v defined on grid edges to the
#     scalar grid on gridpoint centres (same resolution as original)

#     Currently cannot regrid cross-sections of vector quantities --
#     these must be regridded first before taking the cross-section afterwards.
#     (this is because the cross-sections can mess with lat/lon spacing)

#     will need to load a 'dummy' variable to get the coordinates
#     of the gridpoint centres at the same resolution as input data
#     - might be able to just use grids in /g/data instead?
#     -- > these are actually different from the variable grids (?)

#     Inputs:
#         data: input data
#     Outputs:
#         data_regrid: regridded data
#     """

#     sub = identify_subgrid(data)

#     if 'distance' in data.dims:
#         raise Exception(
#             f"Can't horizontally regrid data on '{sub}' grid, regrid to 't' first"
#         )

#     # load the grid to reshape input data to

#     # cut lats/lons of grid to that of input
#     grid = grid.sel(latitude=slice(data.latitude.min(), data.latitude.max()))
#     grid = grid.sel(longitude=slice(data.longitude.min(), data.longitude.max()))

#     # need to squeeze input data (xr.interp_like doesn't like the exta dims for some reason)
#     squeezed_dims = [dim for dim in data.dims if data[dim].size == 1]
#     data = data.squeeze()

#     data_regrid = data.interp_like(grid)

#     # expand dims (any further vertical interpolation needs expanded dims)
#     data_regrid = data_regrid.expand_dims(squeezed_dims)

#     return data_regrid


class DataHolder:
    
    # how much data to (approximately) load at a time, in MB
    memory_limit = 1024
    
    def __init__(self, data, *extra_data):
        self.data = data
        self.data_extra = extra_data
        self.dataset_count = len(extra_data) + 1
        
        self.buffer = 0
        
        self.loaded_data = None
        # self.loaded_data_extra = None
        self.loaded_range = None
        
        self.data_range = (0, len(data.latitude) - 1, 
                           0, len(data.longitude) - 1)
        
        # for centering (filler values)
        self.desired_width = len(data.longitude) // 5
        self.desired_height = len(data.latitude) // 5
        self.centre = None
        
    def set_buffer(self, buffer):
        self.buffer = buffer
        
        # refresh data range just in case (if no other size adjustments will be made)
        if (self.centre):
            self.centre_on_index(*self.centre)
        else:
            self.index_slice(*self.data_range)
        
    def get_data(self, i, load = True, load_cap = None):
        (lat_min_pos, lat_max_pos, long_min_pos, long_max_pos) = self.data_range
        
        loaded = self.loaded_data is not None

        # check that requested data is in range if data is already loaded
        if loaded:
            loaded = i in range(*self.loaded_range)

        # no data loaded - load an appropriate amount to not exceed memory limits (for now 2 GB)
        if not loaded:
            if load:
                # assumed 4 bytes per data entry
                step_size = (lat_max_pos - lat_min_pos) * (long_max_pos - long_min_pos) * 4
                max_steps = (DataHolder.memory_limit * 1024 * 1024) // (step_size * self.dataset_count)

                # set range to start from currently requested position to a max point
                # max is either predefined by argument call, capped by memory or at end of data range
                self.loaded_range = [i, min(i + max_steps, len(self.data.time))]
                if load_cap:
                    self.loaded_range[1] = min(self.loaded_range[1], load_cap)

                # check that non-0 portion will be loaded, otherwise fall through and use default single load
                if self.loaded_range[1] > self.loaded_range[0]:
                    # execute load and force compute
                    self.loaded_data = [data.isel(latitude = slice(lat_min_pos - self.buffer, lat_max_pos + self.buffer), 
                                                  longitude = slice(long_min_pos - self.buffer, long_max_pos + self.buffer),
                                                  time = slice(*self.loaded_range)).load() for data in (self.data, *self.data_extra)]
                    # self.loaded_data = self.data.isel(latitude = slice(lat_min_pos - self.buffer, lat_max_pos + self.buffer), 
                    #                                   longitude = slice(long_min_pos - self.buffer, long_max_pos + self.buffer),
                    #                                   time = slice(*self.loaded_range)).load()
                    # self.loaded_data_extra = (
                    loaded = True

        # even if preloading is disabled, there may already be loaded data which can be used
        if loaded:
            # data_samples = self.loaded_data[i - self.loaded_range[0]]
            data_samples = [item[i - self.loaded_range[0]] for item in self.loaded_data]
        
        else:
            # data_samples = self.data.isel(latitude = slice(lat_min_pos - self.buffer, lat_max_pos + self.buffer), 
            #                               longitude = slice(long_min_pos - self.buffer, long_max_pos + self.buffer),
            #                               time = i)
            data_samples = [data.isel(latitude = slice(lat_min_pos - self.buffer, lat_max_pos + self.buffer), 
                                      longitude = slice(long_min_pos - self.buffer, long_max_pos + self.buffer),
                                      time = i) for data in (self.data, *self.data_extra)]
            
        if self.dataset_count == 1:
            return data_samples[0]
        else:
            return data_samples
    
    def get_data_range(self):
        return self.data_range
    
    # set the boundaries for data (could be from various functions)
    def index_slice(self, lat_min_pos, lat_max_pos, long_min_pos, long_max_pos):
        self.data_range = (self.lat_in_bounds(lat_min_pos), 
                           self.lat_in_bounds(lat_max_pos), 
                           self.long_in_bounds(long_min_pos), 
                           self.long_in_bounds(long_max_pos))
        # self.height = lat_max_pos - lat_min_pos
        # self.width = long_max_pos - long_min_pos
        
    def coord_slice(self, lat_min, lat_max, long_min, long_max):
        point1 = find_nearest_index(self.full_data, (lat_min, long_min))
        point2 = find_nearest_index(self.full_data, (lat_max, long_max))
        self.index_slice(point1[0], point2[0], point1[1], point2[1])     
    
    def lat_in_bounds(self, lat, border = 0):
        return min(max(lat, self.buffer + border), self.get_total_height() - self.buffer - border)
    
    def long_in_bounds(self, long, border = 0):
        return min(max(long, self.buffer + border), self.get_total_width() - self.buffer - border) 
        
    def centre_on_index(self, lat_pos, long_pos):
        half_width = self.desired_width // 2
        half_height = self.desired_height // 2
        
        # hit top or bottom edge
        # lat_pos = max(lat_pos, half_height)
        # lat_pos = min(lat_pos, len(self.data.latitude) - half_height)
        lat_pos = self.lat_in_bounds(lat_pos, half_height)
        
        # hit left or right edge
        # long_pos = max(long_pos, half_width)
        # long_pos = min(long_pos, len(self.data.longitude) - half_width)
        long_pos = self.long_in_bounds(long_pos, half_width)
        
        self.index_slice(lat_pos - half_height, lat_pos + half_height,
                         long_pos - half_width, long_pos + half_width)
        
        self.centre = lat_pos, long_pos
        
    def centre_on_coords(self, lat, long):
        self.centre_on_index(*find_nearest_index(self.data, (lat, long)))
    
    # size setting applies only when centering
    def set_width(self, width):
        if (width < 1):
            # to do: check whole
            raise ValueError("width must be a positive whole number")
        self.desired_width = min(width, len(self.data.longitude))
        
    def get_width(self):
        return self.data_range[3] - self.data_range[2]
    
    def get_total_width(self):
        return len(self.data.longitude)
        
    def set_height(self, height):
        if (height < 1):
            # to do: check whole
            raise ValueError("height must be a positive whole number")
        self.desired_height = min(height, len(self.data.latitude))
        
    def get_height(self):
        return self.data_range[1] - self.data_range[0]
    
    def get_total_height(self):
        return len(self.data.latitude)
        
    def set_size(self, width, height):
        self.set_width(width)
        self.set_height(height)
        
    def get_size(self):
        return self.get_width(), self.get_height()
    
    def get_total_time(self):
        return len(self.data.time)
    
    def get_timestamp(self, i):
        return self.data.time.values[i]
        
    # helper function to use common sizes (all at 16:9 width:height ratio)
    def set_size_preset(self, preset):
        self.set_size(*get_preset_size(preset))
    
class VortHolder(DataHolder):
    def __init__(self, uwind, vwind):
        super().__init__(uwind, vwind)
        self.set_buffer(3)
        
        res = identify_resolution(uwind)
        self.grid = load_var(resolution=res, stream="fx", variable="lnd_mask").load()
    
    def get_data(self, i, load = True, load_cap = None):
        uwind, vwind = super().get_data(i, load, load_cap)
        
        u10 = regrid_vector(uwind, self.grid)
        v10 = regrid_vector(vwind, self.grid)
        
        min_point = find_nearest_index(u10, (self.data.latitude.values[self.get_data_range()[0]], self.data.longitude.values[self.get_data_range()[2]]))
        
        u10 = u10.isel(latitude = slice(min_point[0], min_point[0] + self.get_height()),
                       longitude = slice(min_point[1], min_point[1] + self.get_width())) 
        
        u10, v10 = xr.align(u10, v10)
        
        R_e = 6371e3 # radius of earth

        lats = u10.latitude
        lats_rad = lats * np.pi/180
        # lons = u10.latitude
        # lons_rad = lons * np.pi/180

        # 180/pi comes out when converting degrees to radians
        dv_dl = v10.differentiate('longitude') * 180/np.pi

        rv = u10 * np.cos(lats_rad)
        rv = -rv.differentiate('latitude') * 180/np.pi
        rv += dv_dl
        rv *= 1/(R_e * np.cos(lats_rad))

        return rv 
        
    
class PathDataHolder:
    def __init__(self, data_holder, path):
        self.data_holder = data_holder
        self.path = path
        self.data_bounds = self.find_path_limits()
        self.current_bounds = None
        
        # just to return later - 
        self.data_range = None
        
        # default values
        self.width = data_holder.desired_width
        self.height = data_holder.desired_height
        
        self.loaded_range = None
        
    def set_width(self, width):
        if (width < 1):
            # to do: check whole
            raise ValueError("width must be a positive whole number")
        self.width = width
        
    def get_width(self):
        return self.width
        
    def set_height(self, height):
        if (height < 1):
            # to do: check whole
            raise ValueError("height must be a positive whole number")
        self.height = height
        
    def get_height(self):
        return self.height
        
    def set_size(self, width, height):
        self.set_width(width)
        self.set_height(height)
        
    def get_size(self):
        return self.get_width(), self.get_height()
    
    def get_total_time(self):
        return min(self.data_holder.get_total_time(), len(self.path))
    
    def get_timestamp(self, i):
        return self.data_holder.get_timestamp(i)
        
    # helper function to use common sizes (all at 16:9 width:height ratio)
    def set_size_preset(self, preset):
        self.set_size(*get_preset_size(preset))
        
    def find_path_limits(self, start = 0, stop = -1):
        path_sample = self.path[start:stop]
        min_lat = path_sample[0][0]
        min_long = path_sample[0][1]
        max_lat, max_long = min_lat, min_long
        for item in path_sample:
            min_lat = min(min_lat, item[0])
            max_lat = max(max_lat, item[0])
            min_long = min(min_long, item[1])
            max_long = max(max_long, item[1])
        return min_lat, max_lat, min_long, max_long
    
    def get_data_range(self):
        return self.data_range
        
    def get_data(self, i, load = True, load_cap = None):
        # get desired positions from path and size
        half_width = self.width // 2
        half_height = self.height // 2
        
        # hit top or bottom edge
        # lat_pos = max(self.path[i][0], half_height)
        # lat_pos = min(lat_pos, self.dataHolder.get_total_height() - half_height)
        lat_pos = self.data_holder.lat_in_bounds(self.path[i][0], half_height)
        
        # hit left or right edge
        # long_pos = max(self.path[i][1], half_width)
        # long_pos = min(long_pos, self.dataHolder.get_total_width() - half_width)
        long_pos = self.data_holder.long_in_bounds(self.path[i][1], half_width)
        
        lat_min_pos = lat_pos - half_height
        lat_max_pos = lat_pos + half_height
        long_min_pos = long_pos - half_width
        long_max_pos = long_pos + half_width
        
        loaded = self.loaded_range is not None

        # check that requested data is in range if data is already loaded
        if loaded:
            loaded = i in range(*self.loaded_range)
        
        # load data if required
        if not loaded:
            if load:
                # assumed 4 bytes per data entry
                # this is rough estimation of how much memory each timestep would take and is certainly not accurate
                step_size = (self.data_bounds[1] - self.data_bounds[0] + self.height) * (self.data_bounds[3] - self.data_bounds[2] + self.width) * 4
                max_steps = 2147483648 // step_size

                # set range to start from currently requested position to a max point
                # max is either predefined by argument call, capped by memory or at end of data range
                self.loaded_range = [i, min(i + max_steps, self.data_holder.get_total_time())]
                if load_cap:
                    self.loaded_range[1] = min(self.loaded_range[1], load_cap)
                else:
                    load_cap = self.loaded_range[1]
                    
                # slice data to what will actually be required for the upcoming path steps
                # this is larger than the set size and will be trimmed per timestep
                bounds = self.find_path_limits(*self.loaded_range)
                self.current_bounds = (self.data_holder.lat_in_bounds(bounds[0], half_height) - half_height, 
                                       self.data_holder.lat_in_bounds(bounds[1], half_height) + half_height,
                                       self.data_holder.long_in_bounds(bounds[2], half_width) - half_width, 
                                       self.data_holder.long_in_bounds(bounds[3], half_width) + half_width)
                self.data_holder.index_slice(*self.current_bounds)
                loaded = True
            
        # can use preloaded data even if loading further is disabled
        # if preloading is enabled this is the call that actually forces it to happen for the DataHolder
        if loaded:
            uncut_data = self.data_holder.get_data(i, load, load_cap)
            
            # trim
            data_sample = uncut_data.isel(latitude = slice(lat_min_pos - self.current_bounds[0], lat_max_pos - self.current_bounds[0]),
                                          longitude = slice(long_min_pos - self.current_bounds[2], long_max_pos - self.current_bounds[2]))
        
        # not loaded, no preloading
        else:
            self.data_holder.index_slice(lat_min_pos, lat_max_pos, long_min_pos, long_max_pos)
            data_sample = self.data_holder.get_data(i, load, load_cap)
        self.data_range = lat_min_pos, lat_max_pos, long_min_pos, long_max_pos
        return data_sample


class AnimFeature:
    def __init__(self, priority):
        self.priority = priority
        
    def process_image(self, image, data_range, time):
        return image
    
    def __lt__(self, other):
        return self.priority < other.priority
    
    def __eq__(self, other):
        return self.priority == other.priority
    

class AddBorder(AnimFeature):
    def __init__(self, name = "", image = None, colour = "black", priority = 5):
        super().__init__(priority)
        
        if name and image:
            raise TypeError("Only one of name or image should be provided")
        if not name and not image:
            raise TypeError("One of name or image must be provided")
        if name:
            # prevent PIL from raising image bomb warning
            PIL.Image.MAX_IMAGE_PIXELS = 139249476 
            image = PIL.Image.open(f'Aus400/borders/{name}.png')
            
        self.colour = colour
            
        self.border_mask = image.convert("1")
        self.border_cropped = None
        self.current_data_range = []
        
    def get_cropped(self, data_range):
        # check if already cropped appropriately
        if self.current_data_range != data_range:
            height = self.border_mask.height
            (lat_min_pos, lat_max_pos, long_min_pos, long_max_pos) = data_range
            self.border_cropped = self.border_mask.crop((long_min_pos, height - lat_max_pos, 
                                                         long_max_pos + 1, height - lat_min_pos + 1))
            self.current_data_range = data_range
            
        return self.border_cropped
    
    def process_image(self, image, data_range, time):
        mask = self.get_cropped(data_range)
        image.paste(self.colour, mask = mask)
        return image
            

class AddTime(AnimFeature):
    def __init__(self, font_path = "", size = 40, pos = (40, 40), colour = "black", stroke_width = 3, stroke_fill = "white", priority = 5):
        super().__init__(priority)
        if font_path:
            self.font = ImageFont.truetype(font_path, size)
        else:
            self.font = ImageFont.load_default()
        self.pos = pos
        self.colour = colour
        self.stroke_width = stroke_width
        self.stroke_fill = stroke_fill
        
    def process_image(self, image, data_range, time):
        draw = ImageDraw.Draw(image)
        draw.text(self.pos, str(np.datetime_as_string(time, unit='m')).replace("T", " ") + " UTC", 
                  fill = self.colour, 
                  font = self.font, 
                  stroke_width = self.stroke_width, 
                  stroke_fill = self.stroke_fill)
        return image      
            

                 
class Animator:
    def __init__(self, data_holder, vmin, vmax):
        self.data_holder = data_holder
        self.vmin = vmin
        self.vmax = vmax
        self.im_list = []
        self.features = []
        if (vmax < vmin):
            raise ValueError("vmin cannot be greater than vmax")
        
    
    # cmap related methods
        
    def set_cmap(self, cmap):
        self.cmap = cmap
        
    def load_cmap(self, cmap_name):
        self.set_cmap(plt.get_cmap(cmap_name))
        
    def add_feature(self, feature):
        self.features.append(feature)
        self.features.sort()
        
    #draw
      
    def draw_image(self, i, preload = False, preload_cap = None):
        data = self.data_holder.get_data(i, preload, preload_cap)
        image = set_image_mpl_cmap(field_to_image(data, vmin = self.vmin, vmax = self.vmax), self.cmap)
        
        # check for processing issue
        if image.size != self.data_holder.get_size():
            raise ValueError(f'Wrong size at time {i}')
            
        for feature in self.features:
            image = feature.process_image(image, self.data_holder.get_data_range(), self.data_holder.get_timestamp(i))
            
        return image.convert('RGBA')
    
    def animate(self, save_name, length = 0, start_point = 0, frame_duration = 40):
        if length == 0:
            length = self.data_holder.get_total_time() - start_point
        
        self.im_list = [None] * length
        
        for i in range(length):
            
            # check not out of time bounds
            if i >= self.data_holder.get_total_time():
                break

            self.im_list[i] = self.draw_image(i + start_point, True, start_point + length)
                
        self.im_list[0].save(f'{save_name}.gif',
                       append_images = [im for im in self.im_list[1:] if im is not None], save_all = True, loop = 0, duration = frame_duration)
        
        return
