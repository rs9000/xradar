#!/usr/bin/env python
# Copyright (c) 2022, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Datamet Data I/O
^^^^^^^^^^^^^^^^

This sub-module contains the Datamet xarray backend for reading data from Leonardo's
Datamet data formats into Xarray structures as well as a reader to create a complete
datatree.Datatree.


Example::

    import xradar as xd
    dtree = xd.io.open_datamet_datatree(filename)

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}

"""

__all__ = [
    "DatametBackendEntrypoint",
    "open_datamet_datatree",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import datetime as dt
from collections import OrderedDict

import numpy as np
import xarray as xr
from datatree import DataTree

from xarray.backends.common import AbstractDataStore, BackendArray, BackendEntrypoint, BACKEND_ENTRYPOINTS
from xarray.backends.file_manager import CachingFileManager
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.utils import FrozenDict
from xarray.core.variable import Variable

from ... import util
from ...model import (
    get_altitude_attrs,
    get_azimuth_attrs,
    get_elevation_attrs,
    get_latitude_attrs,
    get_longitude_attrs,
    get_range_attrs,
    get_time_attrs,
    moment_attrs,
    sweep_vars_mapping,
)
from .common import _attach_sweep_groups
from .odim import _assign_root

#: mapping of datamet moment names to CfRadial2/ODIM names
datamet_mapping = {
    "UZ": "DBTH",
    "CZ": "DBZH",
    "V": "VRADH",
    "W": "WRADH",
    "ZDR": "ZDR",
    "KDP": "UKDP",
    "CKDP": "KDP",
    "PHIDP": "PHIDP",
    "SQI": "SQIH",
    "SNR": "SNR",
    "RHOHV": "RHOHV",
    "Quality": "QIND",
    "ClutterMask": "CPA"
}

xradar_mapping = {
    "DBTH": "DBTH",
    "DBZH": "reflectivity",
    "VRADH": "velocity",
    "WRADH": "spectrum_width",
    "ZDR": "differential_reflectivity",
    "UKDP": "UKDP",
    "KDP": "KDP",
    "PHIDP": "differential_phase",
    "SQIH": "SQIH",
    "SNR": "SNR",
    "RHOHV": "cross_correlation_ratio",
    "Quality": "QIND",
    "CPA": "CPA"
}

import os
from PIL import Image


def find_key(key, dictionary):
    for k, v in dictionary.items():
        if k == key:
            return v
        elif isinstance(v, dict):
            esito = find_key(key, v)
            if esito is not None:
                return esito
            else:
                continue
        elif isinstance(v, list):
            for d in v:
                if isinstance(d, dict):
                    esito = find_key(key, d)
                    if esito is not None:
                        return esito
                    else:
                        continue


class _DatametReader:
    def __init__(self, filename):
        self._root = OrderedDict()
        self._filename = filename
        self._measure = set()
        self._sweep = OrderedDict()
        self._pathname = os.path.dirname(filename)
        self._lores = False
        self.datamet2cf(self._pathname)

    @property
    def lores(self):
        return self._lores

    @property
    def filename(self):
        return self._filename

    @property
    def measure(self):
        return self._measure

    @property
    def sweep(self):
        return self._sweep

    @property
    def root(self):
        return self._root

    @property
    def origin(self):
        origin = find_key('ORIGIN', self._root)
        return origin

    def read_graphic(self, filename):
        try:
            image = Image.open(filename)
        except:
            image = None
        return image

    def read_strings(self, pathname=None):
        ret = []
        if not os.path.exists(pathname):
            return ret
        with open(pathname) as f:
            line = f.readline()
            while line:
                ret.append(line)
                line = f.readline()
        return (ret)

    def read_wind(self, dirname, speedname):

        if not os.path.exists(dirname) or not os.path.exists(speedname):
            return (None)

        if os.path.exists(dirname):
            vdir = np.loadtxt(dirname)
        else:
            return None

        if os.path.exists(speedname):
            vspeed = np.loadtxt(speedname)
        else:
            return None
        outw = np.column_stack((vdir, vspeed))
        return (outw)

    def read_array(self, pathname, type: np.dtype, dim, format):
        array = None
        if not os.path.exists(pathname):
            return 0
        if format == '':
            if pathname.suffix.upper() == 'TXT': format = 'txt'

        others = 1
        if format.upper == 'TXT':
            array = self.read_strings(pathname)
        else:
            if format.upper == 'WIND':
                array = self.read_wind(pathname)
            else:
                if format.upper == 'COORDS':
                    array = self.read_coords(pathname, dim)
                else:
                    if format == '' or format.upper == 'DAT' or format.upper == 'AUX' or format.upper == 'RAW':
                        array = np.fromfile(pathname, dtype=type)
                        format = 'dat'
                    else:
                        others = 0
        if others == 0:
            array: Image = self.read_graphic(pathname)
        return array

    def read_coords(self, latname, lonname, altname):

        if not os.path.exists(latname) or not os.path.exists(lonname):
            return (None)

        vlat = np.loadtxt(latname)
        vlon = np.loadtxt(lonname)
        if vlat.size != vlon.size:
            return None

        valt = None
        if os.path.exists(altname):
            valt = np.loadtxt(altname)
            if valt.size != vlat.size:
                valt = None
        if valt is None:
            outa = np.column_stack((vlat, vlon))
        else:
            outa = np.column_stack((vlat, vlon, valt))

        return (outa)

    def LoadValues(self, path, filename):
        if not len(filename) > 0:
            return None
        pathname = os.path.join(path, filename)
        if not os.path.exists(pathname):
            pathname = os.path.join(path, filename.upper())
        if not os.path.exists(pathname):
            filename, ext = filename.split(".")
            pathname = os.path.join(path, filename + "." + ext.lower())
        if not os.path.exists(pathname):
            return None
        val_array = np.loadtxt(pathname)
        return val_array

    def LoadTokens(self, path, filename):
        ret = {}
        pathname = os.path.join(path, filename)
        if not os.path.exists(pathname):
            pathname = os.path.join(path, filename.upper())
        if not os.path.exists(pathname):
            return ret
        data = self.read_strings(pathname)
        for elem in data:
            lvals = elem.split('=')
            if len(lvals) == 2:
                k1: str = str(lvals[0]).upper().strip()
                v1: str = str(lvals[1]).strip()
                if not k1 == 'INHERITS':
                    v1 = v1.upper()
                if k1 in ret.keys():
                    ret[k1].append(v1)
                else:
                    ret[k1] = [v1]
        return (ret)

    def MergeTokens(self, d_in: dict, d_add: dict):
        for k in d_add.keys():
            if k in d_in.keys():
                d_in[k] = d_in[k] + d_add[k]
            else:
                d_in[k] = d_add[k]

    def typeIDL2NP(self, idl_code):
        if idl_code == 1:
            return np.ubyte
        if idl_code == 2:
            return np.int16
        if idl_code == 12:
            return np.uint16
        if idl_code == 3:
            return np.int32
        if idl_code == 13:
            return np.uint32
        if idl_code == 4:
            return np.float32
        if idl_code == 5:
            return np.float64
        return None

    def get_root(self, pathn=None):
        _root = OrderedDict()
        if pathn == None:
            pathn = self._pathname
        files = [f for f in os.listdir(pathn) if os.path.isfile(os.path.join(pathn, f))]
        for f in files:
            self.get_dtm_tree(pathn, f, _root)
        return (_root)

    def get_values(self, content: dict, idx):
        _root = OrderedDict()
        _root['NAVIGATION'] = content['NAVIGATION']
        _root['CALIBRATION'] = content['CALIBRATION']
        _root['GENERIC'] = content['GENERIC']
        _root['SCAN'] = {}
        if 'ELEVATION' in content['SCAN'].keys():
            azoff = float(content['NAVIGATION']['AZOFF'])
            azres = float(content['NAVIGATION']['AZRES'])
            numaz = int(content['GENERIC']['NLINES'])
            _root['SCAN']['ELEVATION'] = np.array(
                [content['SCAN']['ELEVATION'][idx]] * int(content['GENERIC']['NLINES']))
            _root['SCAN']['AZIMUTH'] = np.array([azoff + (azres * x) for x in range(numaz)])
            _root['SCAN']['SCAN_TYPE'] = 'azimuth'
        else:
            if 'AZIMUTH' in content['SCAN'].keys():
                hoff = float(content['NAVIGATION']['HOFF'])
                hres = float(content['NAVIGATION']['HRES'])
                numele = int(content['GENERIC']['NLINES'])
                _root['SCAN']['AZIMUTH'] = np.array(
                    [content['SCAN']['AZIMUTH'][idx]] * int(content['GENERIC']['NLINES']))
                _root['SCAN']['ELEVATION'] = np.array([hoff + (hres * x) for x in range(numele)])
                _root['SCAN']['SCAN_TYPE'] = 'rhi'
        _root['SCAN']['dataset'] = content['SCAN']['dataset'][idx]
        return (_root)

    def get_dtm_tree(self, dirpath, filename, sw):
        upfile = str(filename).upper()
        if upfile.endswith('.TXT'):
            grpname_r = upfile.split('.TXT')[0]
            if grpname_r not in sw:
                sw[grpname_r] = {}
            din: dict = self.LoadTokens(dirpath, filename)
            if 'INHERITS' in din.keys():
                din_inh: dict = self.LoadTokens(dirpath, din['INHERITS'][0])
                self.MergeTokens(din, din_inh)
            for ikey in din.keys():
                if len(din[ikey]) == 1:
                    if ikey != 'INHERITS':
                        sw[grpname_r][ikey] = din[ikey][0]
                if len(din[ikey]) > 1:
                    count = 0
                    for elist in din[ikey]:
                        count += 1
                        sw[grpname_r][ikey + '.' + str(count)] = elist
            if not bool(sw[grpname_r]):
                del sw[grpname_r]
        if upfile.endswith('.DAT') or upfile.endswith('AUX') or upfile.endswith('RAW'):
            grpname_r = upfile.split('.')[0]
            if grpname_r not in sw:
                sw[grpname_r] = {}
            myg = sw[grpname_r]
            din: dict = self.LoadTokens(dirpath, 'generic.txt')
            type_val = din.get('TYPE', ['1'])
            type_code: np.dtype = self.typeIDL2NP(int(type_val[0]))
            assert (type_code is not None)
            nLines = int(din.get('NLINES', ['0'])[0])
            nCols = int(din.get('NCOLS', ['0'])[0])
            nPlanes = int(din.get('NPLANES', ['1'])[0])
            assert (nLines > 0 and nCols > 0)
            din2: dict = self.LoadTokens(dirpath, 'navigation.txt')
            azfile = din2.get('AZFILE', [''])[0]
            elfile = din2.get('ELFILE', [''])[0]
            el_array = self.LoadValues(dirpath, elfile)
            if type(el_array) == np.ndarray:
                myg[elfile.split('.')[0]] = el_array
            elif 'ELOFF' in din2.keys():
                el_array = np.array([float(din2['ELOFF'][0])] * nLines)
                myg['ELEVATION'] = el_array
            az_array = self.LoadValues(dirpath, azfile)
            if type(az_array) == np.ndarray:
                myg[azfile.split('.')[0]] = az_array
            elif 'AZOFF' in din2.keys():
                az_array = np.array([float(din2['AZOFF'][0])] * nLines)
                myg['AZIMUTH'] = az_array
            array = np.fromfile(os.path.join(dirpath, filename), type_code)
            if nPlanes > 1:
                if type(el_array) == np.ndarray:
                    assert (el_array.size == nPlanes)
                    myg[elfile.split('.')[0]] = el_array
                else:
                    if type(az_array) == np.ndarray:
                        assert (az_array.size == nPlanes)
                        myg[azfile.split('.')[0]] = az_array
                rs_a = array.reshape((nPlanes, nLines, nCols))
            else:
                rs_a = array.reshape((nLines, nCols))
            myg['dataset'] = rs_a
        if upfile.endswith('.COORDS'):
            assert ('TBD' == 'TODO')
        if upfile.endswith('.WIND'):
            assert ('TBD' == 'TODO')

    def datamet2cf(self, inpath):
        self._root = self.get_root(inpath)
        self._lowres = 'NOMINAL_DATE' in self._root['GENERIC'].keys()
        for _, dirnames, _ in os.walk(inpath):
            for dirname in dirnames:
                if dirname in datamet_mapping:
                    self._measure.add(dirname)
                    if self._lowres:
                        content = self.get_root(os.path.join(inpath, dirname))
                        sweep_num = int(content['GENERIC']['NPLANES'])
                        for idx in range(sweep_num):
                            swname = "sweep_" + str(idx)
                            if swname not in self._sweep:
                                self._sweep[swname] = {}
                            self._sweep[swname][dirname] = self.get_values(content, idx)
                    else:
                        for _, sweep_num_p, _ in os.walk(os.path.join(inpath, dirname)):
                            for item in sweep_num_p:
                                sweep_num = int(item)
                                swname = "sweep_" + str(sweep_num - 1)
                                if swname not in self._sweep:
                                    self._sweep[swname] = {}
                                self._sweep[swname][dirname] = self.get_root(os.path.join(inpath, dirname, item))


class DatametFile():
    """DatametFile class"""

    def __init__(self, filename, **kwargs):
        self._filename = filename
        self._fp = None

        if isinstance(filename, str):
            self._data = _DatametReader(filename)
        else:
            raise TypeError(
                "Datamet reader currently doesn't support file-like objects"
            )

    def close(self):
        if self._fp is not None:
            self._fp.close()

    @property
    def data(self):
        return self._data

    @property
    def filename(self):
        return self._filename

    @property
    def first_dimension(self):
        if 'ARCHIVIATION' in self._data.root:
            if self._data.root['ARCHIVIATION']['SCAN_TYPE'] == "VOLUMETRICA":
                return "azimuth"
            else:
                return "rhi"
        return None

    @property
    def slices(self):
        slices = self._data.sweep
        return slices


class DatametArrayWrapper(BackendArray):
    """Wraps array of DATAMET data. name=moment var=sweep"""

    def __init__(self, data,
                 ):
        self.data = data
        self.shape = data.shape
        self.dtype = data.dtype

    def __getitem__(self, key: tuple):
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.OUTER_1VECTOR,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple):
        return self.data[key]


class DatametStore(AbstractDataStore):
    """Store for reading DATAMET sweeps via wradlib."""

    def __init__(self, manager, group=None):
        self._manager = manager
        self._group = group
        self._filename = os.path.join(self.filename, 'generic.txt') if os.path.isdir(self.filename) else self.filename
        self._need_time_recalc = False

    @classmethod
    def open(cls, filename, mode="r", group=None, **kwargs):
        if os.path.isdir(filename):
            filename = os.path.join(filename, 'generic.txt')
        manager = CachingFileManager(DatametFile, filename, mode=mode, kwargs=kwargs)
        return cls(manager, group=group)

    @property
    def filename(self):
        with self._manager.acquire_context(False) as root:
            return root.filename

    @property
    def root(self):
        with self._manager.acquire_context(False) as root:
            return root

    def _acquire(self, needs_lock=True):
        with self._manager.acquire_context(needs_lock) as root:
            ds = root.slices[self._group]
        return ds

    @property
    def ds(self):
        return self._acquire()

    @property
    def origin(self):
        return self.root.data.origin

    def open_store_variable(self, measure, var):
        dim = self.root.first_dimension
        if dim is None:
            dim = var['SCAN']['SCAN_TYPE']

        raw = var['SCAN']['dataset']
        name = measure

        data = indexing.LazilyOuterIndexedArray(DatametArrayWrapper(raw))
        encoding = {"group": self._group, "source": self._filename}

        mname = datamet_mapping.get(name, name)
        mapping = sweep_vars_mapping.get(mname, {})
        attrs = {key: mapping[key] for key in moment_attrs if key in mapping}
        attrs["add_offset"] = float(var['CALIBRATION']['OFFSET'])
        attrs["scale_factor"] = float(var['CALIBRATION']['SLOPE']) if 'SLOPE' in var['CALIBRATION'] else 1
        attrs["_Undetect"] = int(var['CALIBRATION']['VOIDIND']) if 'VOIDIND' in var['CALIBRATION'] else 0
        attrs["_FillValue"] = int(var['CALIBRATION']['NULLIND']) if 'NULLIND' in var['CALIBRATION'] else -1
        attrs[
            "coordinates"
        ] = "elevation azimuth range latitude longitude altitude time"  ##time
        return {mname: Variable((dim, "range"), data, attrs, encoding)}

    def open_store_coordinates(self, var):
        firstdim = self.root.first_dimension
        if firstdim is None:
            firstdim = var['SCAN']['SCAN_TYPE']
        sweep_mode = "azimuth_survelliance" if firstdim == "azimuth" else "rhi"
        sweep_number = int(self._group[-1])
        prt_mode = "not_set"
        follow_mode = "not_set"
        lon = float(var['NAVIGATION']['ORIG_LON'])
        lat = float(var['NAVIGATION']['ORIG_LAT'])
        alt = float(var['NAVIGATION']['ORIG_ALT'])

        num_bin = int(var['GENERIC']['NCOLS'])
        num_rays = int(var['GENERIC']['NLINES'])

        if 'DATE' in var['GENERIC'].keys() and 'TIME' in var['GENERIC'].keys():
            timestr = f"{var['GENERIC']['DATE']}T{var['GENERIC']['TIME']}Z"  # hh:mm
            time = dt.datetime.strptime(timestr, "%d-%m-%YT%H:%MZ")
            time2 = dt.datetime(time.year, time.month, time.day, time.hour, time.second, time.microsecond,
                                tzinfo=dt.timezone(dt.timedelta(0)))

            raytimes = np.array(
                [time2.timestamp() for x in range(num_rays)]
            )
        else:
            if 'PPITIME' in var['GENERIC'] and 'NOMVEL' in var['GENERIC']:
                timestr = f"{var['GENERIC']['PPITIME']}"
                rotspeed = float(var['GENERIC']['NOMVEL'])
                if rotspeed < 0:
                    rotspeed *= -1

                time = dt.datetime.strptime(timestr, "%S.%M.%H-%d.%m.%y")
                time2 = dt.datetime(time.year, time.month, time.day, time.hour, time.minute, time.second,
                                    tzinfo=dt.timezone(dt.timedelta(0)))

                angle_resol = float(var['NAVIGATION']['AZRES']) if firstdim == "azimuth" else float(
                    var['NAVIGATION']['ELRES'])
                raytime = angle_resol / rotspeed

                raytimes = np.array(
                    [time2.timestamp() + (x * raytime) for x in range(num_rays)]
                )
            else:
                timestr = f"{self.root.data.root['GENERIC']['DATE']}T{self.root.data.root['GENERIC']['TIME']}Z"
                time = dt.datetime.strptime(timestr, "%Y-%m-%dT%H:%MZ")
                time2 = dt.datetime(time.year, time.month, time.day, time.hour, time.second, time.microsecond,
                                    tzinfo=dt.timezone(dt.timedelta(0)))

                raytimes = np.array(
                    [time.timestamp() for x in range(num_rays)]
                )

        dims = ("azimuth", "elevation")
        if firstdim == dims[1]:
            dims = (dims[1], dims[0])

        angle = var['SCAN'][dims[1].upper()][0]
        if angle > 90 and firstdim == "azimuth":
            angle = np.round(angle - 360., 2)

        start_range = float(var['NAVIGATION']['RANGEOFF'])
        range_step = float(var['NAVIGATION']['RANGERES'])

        stop_range = start_range + range_step * num_bin

        rng = np.arange(
            start_range + range_step / 2,
            stop_range + range_step / 2,
            range_step,
            dtype="float32",
        )[:num_bin]

        range_attrs = get_range_attrs()
        range_attrs["meters_between_gates"] = range_step
        range_attrs["spacing_is_constant"] = "true"
        range_attrs["meters_to_center_of_first_gate"] = rng[0]

        dim = dims[0]
        encoding = {"group": self._group}

        rtime_attrs = get_time_attrs("1970-01-01T00:00:00Z")

        rng = Variable(("range",), rng, range_attrs)
        azimuth = var['SCAN']['AZIMUTH']
        azimuth = Variable((dim,), azimuth, get_azimuth_attrs(), encoding)
        elevation = var['SCAN']['ELEVATION']
        elevation = Variable((dim,), elevation, get_elevation_attrs(), encoding)
        time = Variable((dim,), raytimes, rtime_attrs, encoding)

        coordinates = {
            "azimuth": azimuth,
            "elevation": elevation,
            "time": time,
            "range": rng,
            "sweep_mode": Variable((), sweep_mode),
            "sweep_number": Variable((), sweep_number),
            "prt_mode": Variable((), prt_mode),
            "follow_mode": Variable((), follow_mode),
            "sweep_fixed_angle": Variable((), angle),
            "longitude": Variable((), lon, get_longitude_attrs()),
            "latitude": Variable((), lat, get_latitude_attrs()),
            "altitude": Variable((), alt, get_altitude_attrs()),
        }

        return coordinates

    def get_variables(self):
        return FrozenDict(
            (k1, v1)
            for k, v in self.ds.items()
            for k1, v1 in {
                **self.open_store_variable(k, v),
                **self.open_store_coordinates(v),
            }.items()
        )

    def get_attrs(self):
        attributes = {}
        attributes['origin'] = self.origin
        return FrozenDict(attributes)


class DatametBackendEntrypoint(BackendEntrypoint):
    """Xarray BackendEntrypoint for Datamet data."""

    description = "Open Datamet files in Xarray"
    url = ""

    def open_dataset(
        self,
        filename_or_obj,
        *,
        mask_and_scale=True,
        decode_times=True,
        concat_characters=True,
        decode_coords=True,
        drop_variables=None,
        use_cftime=None,
        decode_timedelta=None,
        group=None,
        reindex_angle=False,
        first_dim="auto",
        site_coords=True,
    ):
        store = DatametStore.open(
            filename_or_obj,
            group=group,
            loaddata=False,
        )

        store_entrypoint = StoreBackendEntrypoint()

        ds = store_entrypoint.open_dataset(
            store,
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            concat_characters=concat_characters,
            decode_coords=decode_coords,
            drop_variables=drop_variables,
            use_cftime=use_cftime,
            decode_timedelta=decode_timedelta,
        )

        # reassign azimuth/elevation/time coordinates
        ds = ds.assign_coords({"azimuth": ds.azimuth})
        ds = ds.assign_coords({"elevation": ds.elevation})
        ds = ds.assign_coords({"time": ds.time})

        ds.encoding["engine"] = "datamet"

        # handle duplicates and reindex
        if decode_coords and reindex_angle is not False:
            ds = ds.pipe(util.remove_duplicate_rays)
            ds = ds.pipe(util.reindex_angle, **reindex_angle)
            ds = ds.pipe(util.ipol_time)

        # handling first dimension
        dim0 = "elevation" if ds.sweep_mode.load() == "rhi" else "azimuth"
        if first_dim == "auto":
            if "time" in ds.dims:
                ds = ds.swap_dims({"time": dim0})
            ds = ds.sortby(dim0)
        else:
            if "time" not in ds.dims:
                ds = ds.swap_dims({dim0: "time"})
            ds = ds.sortby("time")

        # assign geo-coords
        if site_coords:
            ds = ds.assign_coords(
                {
                    "latitude": ds.latitude,
                    "longitude": ds.longitude,
                    "altitude": ds.altitude,
                }
            )

        return ds


def _get_datamet_group_names(inpath):
    for _, dirnames, _ in os.walk(inpath):
        for dirname in dirnames:
            if dirname in datamet_mapping:
                try:
                    cnt = len(np.loadtxt(os.path.join(inpath, dirname, 'ELEVATION.txt')))  ##Low resolution
                except:
                    cnt = len(next(os.walk(os.path.join(inpath, dirname)))[1])
                return [f"sweep_{i}" for i in range(cnt)]
    return None


def open_datamet_datatree(filename_or_obj, **kwargs):
    """Open DATAMET dataset as :py:class:`datatree.DataTree`.

    Parameters
    ----------
    filename_or_obj : str, Path, file-like or DataStore
        Strings and Path objects are interpreted as a path to a local or remote
        radar file

    Keyword Arguments
    -----------------
    sweep : int, list of int, optional
        Sweep number(s) to extract, default to first sweep. If None, all sweeps are
        extracted into a list.
    first_dim : str
        Can be ``time`` or ``auto`` first dimension. If set to ``auto``,
        first dimension will be either ``azimuth`` or ``elevation`` depending on
        type of sweep. Defaults to ``auto``.
    reindex_angle : bool or dict
        Defaults to False, no reindexing. Given dict should contain the kwargs to
        reindex_angle. Only invoked if `decode_coord=True`.
    fix_second_angle : bool
        If True, fixes erroneous second angle data. Defaults to ``False``.
    site_coords : bool
        Attach radar site-coordinates to Dataset, defaults to ``True``.
    kwargs : dict
        Additional kwargs are fed to :py:func:`xarray.open_dataset`.

    Returns
    -------
    dtree: datatree.DataTree
        DataTree
    """
    if os.path.isdir(filename_or_obj):
        filename_or_obj = os.path.join(filename_or_obj, 'generic.txt')
    # handle kwargs, extract first_dim
    backend_kwargs = kwargs.pop("backend_kwargs", {})
    # first_dim = backend_kwargs.pop("first_dim", None)
    sweep = kwargs.pop("sweep", None)
    sweeps = []
    kwargs["backend_kwargs"] = backend_kwargs

    if isinstance(sweep, str):
        sweeps = [sweep]
    elif isinstance(sweep, int):
        sweeps = [f"sweep_{sweep}"]
    elif isinstance(sweep, list):
        if isinstance(sweep[0], int):
            sweeps = [f"sweep_{i + 1}" for i in sweep]
        else:
            sweeps.extend(sweep)
    else:
        sweeps = _get_datamet_group_names(os.path.dirname(filename_or_obj))

    if sweeps is not None:
        ds = [
            xr.open_dataset(filename_or_obj, group=swp, engine="datamet", **kwargs)
            for swp in sweeps
        ]

        ds.insert(0, xr.Dataset())

        # create datatree root node with required data
        dtree = DataTree(data=_assign_root(ds), name="root")
        # return datatree with attached sweep child nodes
        return _attach_sweep_groups(dtree, ds[1:])
    else:
        return None


BACKEND_ENTRYPOINTS["datamet"] = ("datamet", DatametBackendEntrypoint)
