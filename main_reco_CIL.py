from cil import io, framework, recon
import pydicom
import matplotlib.pyplot as plt
from cil.utilities.display import show_geometry
from cil.utilities.jupyter import islicer
from cil.processors import Padder, RingRemover, AbsorptionTransmissionConverter, TransmissionAbsorptionConverter
import scipy
import numpy as np
from cil.optimisation.operators import BlurringOperator
from cil.utilities import noise
class Config:
    def __init__(self, distance_source_origin, distance_origin_detector, detector_pixel_size,
                 num_projections, angles):
        self.distance_source_origin = distance_source_origin
        self.distance_origin_detector = distance_origin_detector
        self.detector_pixel_size = detector_pixel_size
        self.num_projections = num_projections
        self.angles = angles


fname = 'Pacjent21'
dcm_img = pydicom.read_file('./DANE_SCCS/DANE_SCCS/ORCA/911101088038_20240410/' + fname + '.dcm')
cfg = Config(dcm_img.DistanceSourceToPatient,
             dcm_img.DistanceSourceToDetector,
             dcm_img.ImagerPixelSpacing[0],
             dcm_img.NumberOfFrames,
             dcm_img.PositionerPrimaryAngleIncrement)

reader = io.TIFFStackReader(file_name='./DANE_SCCS/DANE_SCCS/ORCA/911101088038_20240410/'+fname+'/')
data_temp = reader.read()/255
data = np.zeros((data_temp.shape)).astype('float32')

for j in range(0, data_temp.shape[0]):
    I0 = np.mean([np.mean(data_temp[j, :, :10]),
                  np.mean(data_temp[j, :, -10:])])
    img = data_temp[j, :, :]
    img[img > I0] = I0

    img = -np.log(img / I0, where=img > 0)
    data[j, :, :] = img.astype('float32')

data = data_temp
ag = framework.AcquisitionGeometry.create_Cone3D(source_position=[0, cfg.distance_source_origin, 0], \
                                                 detector_position=[0,
                                                                    cfg.distance_origin_detector-cfg.distance_source_origin,
                                                                    0], ) \
    .set_angles(cfg.angles, angle_unit='degree', ) \
    .set_panel([data.shape[1], data.shape[2]], pixel_size=(cfg.detector_pixel_size, cfg.detector_pixel_size))

#ad.RingRemover(decNum=4, wname='db10', sigma=1.5, info=True)

ig = framework.ImageGeometry(voxel_num_x=512,
                             voxel_num_y=512,
                             voxel_num_z=512,
                             voxel_size_x=1,
                             voxel_size_y=1,
                             voxel_size_z=1)


ad = framework.AcquisitionData(data, geometry=ag)
#data_padded = Padder.constant(pad_width=86, constant_values=0)(ad)

fdk = recon.FDK(ad, ig)
fdk.set_filter(filter='ram-lak', cutoff=0.4)
fdk.set_fft_order(10)
fdk.get_filter_array()

out = fdk.run()
scipy.io.savemat('recon_'+fname+'.mat', {'recon': out.array})

#show_geometry(ag, image_geometry=ig)
#plt.imshow(out.array[250, :, :])
#plt.pause(0.01)

