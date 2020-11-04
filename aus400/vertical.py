#!/g/data/hh5/public/apps/nci_scripts/python-analysis3
# Copyright 2020 Scott Wales
# author: Scott Wales <scott.wales@unimelb.edu.au>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .cat import load
from . import xgcm
from .regrid import identify_resolution, identify_subgrid

def vertical_interp(ds, source, target):
    """
    Vertically interpolate the data in ds to the levels of 'target'
    """

    grid = xgcm.grid(ds)
    source = match_slice(source, ds)

    ds = ds.chunk({'model_level_number': None})
    source = source.chunk({'model_level_number': None})

    print(ds)
    print(source)

    return grid.transform(ds, 'Z', target, target_data=source)


def to_plev(ds, levels):
    """
    Interpolate the data in ds to the supplied pressure levels
    """

    res = identify_resolution(ds)
    sub = identify_subgrid(ds)

    if sub != 't':
        raise Exception(f"Can't vertically regrid data on '{sub}' grid, regrid to 't' first")

    pressure = load(resolution=res, stream='mdl', variable='pressure')['pressure']

    return vertical_interp(ds, pressure, levels)


def to_height(ds, levels):
    """
    Interpolate the data in ds to the supplied height levels
    """

    res = identify_resolution(ds)
    sub = identify_subgrid(ds)

    if sub != 't':
        raise Exception(f"Can't vertically regrid data on '{sub}' grid, regrid to 't' first")

    height = load(resolution=res, stream='fx', variable='height_rho')['height_rho']

    return vertical_interp(ds, height, levels)


def match_slice(da, target):
    """
    Match da to the slicing of target
    """
    
    return da.sel({d: target[d] for d in da.dims})


