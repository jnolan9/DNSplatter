import torch
import numpy as np
import json
import os

def readRotFromTransforms(json_path,target_idx):

    with open(json_path) as json_file:
        contents = json.load(json_file)

        frames = contents["frames"]
        for idx, frame in enumerate(frames):

            if idx == target_idx:
                cam_name = frame["file_path"]

                # NeRF 'transform_matrix' is a camera-to-world transform
                c2w = np.array(frame["transform_matrix"])

                # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                c2w[:3, 1:3] *= -1

                # get the world-to-camera transform and set R, T
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
                T = w2c[:3, 3]

    return cam_name,R,T,w2c


def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

checkpoint_path = '/home/jnolan9/DNSplatter/dn-splatter/outputs/phomo_mod_cornelia/dn-splatter/2025-07-24_101326/nerfstudio_models/step-000006999.ckpt'

checkpoint = torch.load(checkpoint_path, map_location="cpu")
data = checkpoint["pipeline"]

dkeys = data.keys()

means = data['_model.gauss_params.means']
normals = data['_model.gauss_params.normals'] # expressed in world
features_dc = data['_model.gauss_params.features_dc']
features_rest = data['_model.gauss_params.features_rest']
opactities = data['_model.gauss_params.opacities']
scales = data['_model.gauss_params.scales']
quats = data['_model.gauss_params.quats']  #Gaussian to world


json_path = '/home/jnolan9/DNSplatter/phomo_mod_cornelia/transforms.json'
target_idx = 0

cam_name,R,T,w2c = readRotFromTransforms(json_path,target_idx)
print(R)
print(T)
print(cam_name)
print(w2c)