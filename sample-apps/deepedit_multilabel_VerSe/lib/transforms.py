# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

logger = logging.getLogger(__name__)

# You can write your transforms here... which can be used in your train/infer tasks
import contextlib

import numpy as np
from monai.transforms import MapTransform, Spacingd
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.utils import allow_missing_keys_mode
from monai.utils import ensure_tuple_rep
from monai.utils.enums import InverseKeys

import json
import logging
from typing import Dict, Hashable, Mapping

import numpy as np
from monai.config import KeysCollection
from monai.transforms import MapTransform, Randomizable
from scipy.ndimage import distance_transform_cdt
from skimage import measure


@contextlib.contextmanager
def _nullcontext(x):
    """
    This is just like contextlib.nullcontext but also works in Python 3.6.
    """
    yield x


class VarSpacingd(MapTransform, InvertibleTransform):
    """By default, scale to 1mm voxels. If image is too big, calculate the scaling factor."""

    def __init__(
        self, keys, mode, min_spacing=1.0, spacing_increment=0.1, mem_thresh=40
    ):
        MapTransform.__init__(self, keys)
        self.min_spacing = min_spacing
        self.mode = mode
        self.spacing_increment = spacing_increment
        self.mem_thresh = mem_thresh

    def __call__(self, data):
        im_shape = np.array(data[self.keys[0]].shape[1:])
        ndim = len(im_shape)
        init_spacing = data[self.keys[0] + "_meta_dict"]["pixdim"][1:4]
        out_spacing = np.array(
            ensure_tuple_rep(self.min_spacing, ndim), dtype=np.float32
        )
        nbyper = sum(data[k].itemsize for k in self.keys)

        # loop until satisfactory size is found
        while True:
            out_shape = im_shape * init_spacing / out_spacing
            out_nvox = np.prod(out_shape)
            # out mem is num vox * nbytes per vox / 1e6 for MB.
            out_mem = out_nvox * nbyper / 1e6
            if out_mem < self.mem_thresh:
                break
            # increase spacing by user-defined increment
            out_spacing += self.spacing_increment
        resampler = Spacingd(self.keys, out_spacing, mode=self.mode)
        d = resampler(data)
        # Add info on voxel spacing
        for key in self.key_iterator(d):
            d[key + InverseKeys.KEY_SUFFIX][-1][InverseKeys.EXTRA_INFO][
                "var_spacing"
            ] = out_spacing
        return d

    def inverse(self, d):
        key = next(self.key_iterator(d))
        var_spacing = d[key + InverseKeys.KEY_SUFFIX][-1][InverseKeys.EXTRA_INFO][
            "var_spacing"
        ]
        resampler = Spacingd(self.keys, var_spacing, mode=self.mode)
        # Change the invertible name from Spacingd to VarSpacingd
        for key in self.key_iterator(d):
            d[key + InverseKeys.KEY_SUFFIX][-1][
                InverseKeys.CLASS_NAME
            ] = resampler.__class__.__name__
            d[key + InverseKeys.KEY_SUFFIX][-1][InverseKeys.ID] = id(resampler)
        context_manager = (
            allow_missing_keys_mode if self.allow_missing_keys else _nullcontext
        )
        with context_manager(resampler):
            return resampler.inverse(d)


class AddInitialSeedPointMissingLabelsd(Randomizable, MapTransform):
    """
    Add random guidance as initial seed point for a given label.
    Note that the label is of size (C, D, H, W) or (C, H, W)
    The guidance is of size (2, N, # of dims) where N is number of guidance added.
    # of dims = 4 when C, D, H, W; # of dims = 3 when (C, H, W)
    Args:
        label: label source.
        guidance: key to store guidance.
        sids: key that represents lists of valid slice indices for the given label.
        sid: key that represents the slice to add initial seed point.  If not present, random sid will be chosen.
        connected_regions: maximum connected regions to use for adding initial points.
    """

    def __init__(
        self,
        keys: KeysCollection,
        guidance: str = "guidance",
        sids: str = "sids",
        sid: str = "sid",
        connected_regions: int = 5,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.sids_key = sids
        self.sid_key = sid
        self.sid: Dict[str, int] = dict()
        self.guidance = guidance
        self.connected_regions = connected_regions

    def _apply(self, label, sid, key_label):
        dimensions = 3 if len(label.shape) > 3 else 2
        self.default_guidance = [-1] * (dimensions + 1)

        dims = dimensions
        if sid is not None and dimensions == 3:
            dims = 2
            label = label[0][..., sid][np.newaxis]  # Assume channel is first and depth is last CHWD

        # import matplotlib.pyplot as plt
        # plt.imshow(label[0])
        # plt.title(f'label as is {key_label}')
        # plt.show()
        # plt.close()

        # THERE MAY BE MULTIPLE BLOBS FOR SINGLE LABEL IN THE SELECTED SLICE
        label = (label > 0.5).astype(np.float32)
        # measure.label: Label connected regions of an integer array - Two pixels are connected
        # when they are neighbors and have the same value
        blobs_labels = measure.label(label.astype(int), background=0) if dims == 2 else label

        pos_guidance = []
        # If there are is presence of that label in this slice
        if np.max(blobs_labels) <= 0:
            pos_guidance.append(self.default_guidance)
        else:
            # plt.imshow(blobs_labels[0])
            # plt.title(f'Blobs {key_label}')
            # plt.show()
            # plt.close()
            for ridx in range(1, 2 if dims == 3 else self.connected_regions + 1):
                if dims == 2:
                    label = (blobs_labels == ridx).astype(np.float32)
                    if np.sum(label) == 0:
                        pos_guidance.append(self.default_guidance)
                        continue

                # plt.imshow(label[0])
                # plt.title(f'Label postprocessed with blob number {key_label}')
                # plt.show()

                # plt.imshow(distance_transform_cdt(label)[0])
                # plt.title(f'Transform CDT {key_label}')
                # plt.show()

                # The distance transform provides a metric or measure of the separation of points in the image.
                # This function calculates the distance between each pixel that is set to off (0) and
                # the nearest nonzero pixel for binary images - http://matlab.izmiran.ru/help/toolbox/images/morph14.html
                distance = distance_transform_cdt(label).flatten()
                probability = np.exp(distance) - 1.0

                idx = np.where(label.flatten() > 0)[0]
                seed = self.R.choice(idx, size=1, p=probability[idx] / np.sum(probability[idx]))
                dst = distance[seed]

                g = np.asarray(np.unravel_index(seed, label.shape)).transpose().tolist()[0]
                g[0] = dst[0]  # for debug
                if dimensions == 2 or dims == 3:
                    pos_guidance.append(g)
                else:
                    # Clicks are created using this convention Channel Height Width Depth (CHWD)
                    pos_guidance.append([g[0], g[-2], g[-1], sid])  # Assume channel is first and depth is last CHWD

        return np.asarray([pos_guidance])

    def _randomize(self, d, key_label):
        sids = d.get(self.sids_key, None).get(key_label, None) if d.get(self.sids_key, None) is not None else None
        sid = d.get(self.sid_key, None).get(key_label, None) if d.get(self.sid_key, None) is not None else None
        if sids is not None and sids:
            if sid is None or sid not in sids:
                sid = self.R.choice(sids, replace=False)
        else:
            logger.info(f"Not slice IDs for label: {key_label}")
            sid = None
        self.sid[key_label] = sid

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "label":
                label_guidances = {}
                for key_label in d["sids"].keys():
                    # Randomize: Select a random slice
                    self._randomize(d, key_label)
                    # Generate guidance base on selected slice
                    tmp_label = np.copy(d[key])
                    # Taking one label to create the guidance
                    if key_label != "background":
                        tmp_label[tmp_label != float(d["label_names"][key_label])] = 0
                    else:
                        tmp_label[tmp_label != float(d["label_names"][key_label])] = 1
                        tmp_label = 1 - tmp_label
                    label_guidances[key_label] = json.dumps(
                        self._apply(tmp_label, self.sid.get(key_label, None), key_label).astype(int).tolist()
                    )
                d[self.guidance] = label_guidances
                return d
            else:
                print("This transform only applies to label key")
        return d


class FindAllValidSlicesMissingLabelsd(MapTransform):
    """
    Find/List all valid slices in the labels.
    Label is assumed to be a 4D Volume with shape CHWD, where C=1.
    Args:
        label: key to the label source.
        sids: key to store slices indices having valid label map.
    """

    def __init__(
        self,
        keys: KeysCollection,
        sids="sids",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.sids = sids

    def _apply(self, label, d):
        sids = {}
        for key_label in d["label_names"].keys():
            l_ids = []
            for sid in range(label.shape[-1]):  # Assume channel is first and depth is last CHWD
                if d["label_names"][key_label] in label[0][..., sid]:
                    l_ids.append(sid)
            # If there are not slices with the label
            if l_ids == []:
                l_ids = [-1] * 10
            sids[key_label] = l_ids
        return sids

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "label":
                label = d[key]
                if label.shape[0] != 1:
                    raise ValueError("Only supports single channel labels!")

                if len(label.shape) != 4:  # only for 3D
                    raise ValueError("Only supports label with shape CHWD!")

                sids = self._apply(label, d)
                if sids is not None and len(sids.keys()):
                    d[self.sids] = sids
                return d
            else:
                print("This transform only applies to label key")
        return d
