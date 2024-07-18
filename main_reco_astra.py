# filename: reconstruct.py
from __future__ import print_function
from __future__ import division

from os import makedirs
from os.path import join, exists
import numpy as np
import pydicom.dicomio as pydcm
from imageio import imread, get_writer
import matplotlib.pyplot as plt
import astra
import scipy
import cv2

class Config:
    def __init__(self, distance_source_origin, distance_origin_detector, detector_pixel_size,
                 num_projections, angles,
                 raw_dir, preproc_dir, proj_dir, reco_dir):
        self.distance_source_origin = distance_source_origin
        self.distance_origin_detector = distance_origin_detector
        self.detector_pixel_size = detector_pixel_size
        self.num_projections = num_projections
        self.angles = angles
        self.raw_dir = raw_dir
        self.preproc_dir = preproc_dir
        self.proj_dir = proj_dir
        self.reco_dir = reco_dir


def save_tiff(name, im_in):
    # Expects image as floating point in range [0, 1].
    im = im_in.copy()
    im = np.round(im * 65535).astype(np.uint16)
    with get_writer(name) as writer:
        writer.append_data(im, {'compress': 0})


multiply_radians = lambda i: i * np.pi / 180
vectorized_multiply_radians = np.vectorize(multiply_radians)

dcm_img = pydcm.read_file('./DANE_SCCS/DANE_SCCS/ORCA/34111506184_20240410/Pacjent10.dcm')
cfg = Config(dcm_img.DistanceSourceToPatient,
             dcm_img.DistanceSourceToDetector,
             dcm_img.ImagerPixelSpacing[0],
             dcm_img.NumberOfFrames,
             vectorized_multiply_radians(dcm_img.PositionerPrimaryAngleIncrement),
             'raw', 'preprocessed', 'projections', 'reconstruction')

if not exists(cfg.reco_dir):
    makedirs(cfg.reco_dir)

detector_pixel_size_in_origin = \
    cfg.detector_pixel_size * cfg.distance_source_origin / \
    (cfg.distance_source_origin + cfg.distance_origin_detector)

# Determine dimensions of projection images.
im = dcm_img.pixel_array
# im = (im-np.min(im))/(np.max(im)-np.min(im))
# im = im[:, 480, :]
dims = im.shape

# dims[0]: Number of rows in the projection image, i.e., the height of the
#          detector. This is Y in the Cartesian coordinate system.
# dims[1]: Number of columns in the projection image, i.e., the width of the
#          detector. This is X in the Cartesian coordinate system.
detector_rows = dims[1]
detector_columns = dims[2]

M = -np.inf
# Load projection images.
projections = np.zeros((detector_rows, cfg.num_projections, detector_columns))
for proj in range(cfg.num_projections):

    img = im[proj,:,:].astype(float)
    #img -= 4095
    #img = abs(img)
    img /= 255#4095

    #I0 = np.mean([np.mean(img[:5, :]),
    #              np.mean(img[-5:, :])])
    #img[img > I0] = I0

    #img = -np.log(img / I0, where=img > 0)
    projections[:, proj, :] = img

    projections /= np.max(projections)

# Copy projection images into ASTRA Toolbox.
"""proj_geom = astra.create_proj_geom('cone', float(dcm_img.ImagerPixelSpacing[0]), float(dcm_img.ImagerPixelSpacing[1]),
                                   detector_rows, detector_columns,
                                   np.asarray(cfg.angles),
                                   float(cfg.distance_source_origin), float(cfg.distance_origin_detector))
"""

proj_geom = astra.create_proj_geom('cone', 1, 1,
                                        detector_rows, detector_columns,
                                        np.asarray(cfg.angles),
                                        (float(cfg.distance_source_origin)+float(cfg.distance_origin_detector)) /
                                        detector_pixel_size_in_origin, 0)


projections_id = astra.data3d.create('-sino', proj_geom, projections)

# vol_geom = astra.creators.create_vol_geom(detector_rows, detector_columns, float(cfg.num_projections))
vol_geom = astra.creators.create_vol_geom(detector_columns, detector_columns, detector_rows)
reconstruction_id = astra.data3d.create('-vol', vol_geom)

alg_cfg = astra.astra_dict('BP3D_CUDA')
alg_cfg['ProjectionDataId'] = projections_id
alg_cfg['ReconstructionDataId'] = reconstruction_id
alg_cfg['ShortScan'] = True
alg_cfg['FilterType'] = 'Ramlak'


# alg_cfg['option']['FilterType'] = 'projection'
# alg_cfg['option']['FilterSinogramId'] = filter_id

algorithm_id = astra.algorithm.create(alg_cfg)
print('Reconstructing... ', end='')
astra.algorithm.run(algorithm_id)
print('done')
print('')
reconstruction = astra.data3d.get(reconstruction_id)
print('Saving...')
scipy.io.savemat('test.mat', {'reconstruction': reconstruction})
print('done')
print('')

