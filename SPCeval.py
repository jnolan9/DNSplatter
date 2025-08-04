"""Classes for storing SPC data.
Author: Travis Driver
"""
import os
from typing import NamedTuple

import numpy as np
import gtsam

from utils.read_write_model import (
    rotmat2qvec,
    qvec2rotmat,
    write_cameras_text,
    read_cameras_text,
    detect_model_format,
)


class SPCImage(NamedTuple):
    """Stores camera extrinsics and Sun vector information."""

    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    svec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    intens: np.ndarray
    point3D_ids: np.ndarray
    scale: float
    bias: float

    def qvec2rotmat(self) -> np.ndarray:
        """Converts quaternion to rotation matrix."""
        return qvec2rotmat(self.qvec)

    def apply_sim3(self, S_AB: gtsam.Similarity3) -> "SPCImage":
        """Apply Sim(3) transformation to stored pose."""
        T_BC = gtsam.Pose3(gtsam.Rot3(self.qvec2rotmat()), self.tvec).inverse()
        T_CA = S_AB.transformFrom(T_BC).inverse()
        return SPCImage(
            id=self.id,
            qvec=rotmat2qvec(T_CA.rotation().matrix()).flatten(),
            tvec=T_CA.translation().flatten(),
            svec=self.svec,
            camera_id=self.camera_id,
            name=self.name,
            xys=self.xys,
            intens=self.intens,
            point3D_ids=self.point3D_ids,
            scale=self.scale,
            bias=self.bias,
        )


class SPCPoint3D(NamedTuple):
    """Stores surface landmark information."""

    id: int
    xyz: int
    nvec: np.ndarray
    albedo: float
    rgb: np.ndarray
    error: float
    image_ids: np.ndarray
    point2D_idxs: np.ndarray

    def apply_sim3(self, S_AB: gtsam.Similarity3) -> "SPCPoint3D":
        """Apply Sim(3) transformation to stored points."""
        return SPCPoint3D(
            id=self.id,
            xyz=S_AB.transformFrom(self.xyz).flatten(),
            nvec=(S_AB.rotation().matrix() @ self.nvec[..., None]).flatten(),
            albedo=self.albedo,
            rgb=self.rgb,
            error=self.error,
            image_ids=self.image_ids,
            point2D_idxs=self.point2D_idxs,
        )


def read_spcimages_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                svec = np.array(tuple(map(float, elems[8:11])))
                camera_id = int(elems[11])
                image_name = elems[12]
                scale = float(elems[13])
                bias = float(elems[14])
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::4])), tuple(map(float, elems[1::4]))])
                intens = np.array(tuple(map(float, elems[2::4])))
                point3D_ids = np.array(tuple(map(int, elems[3::4])))
                images[image_id] = SPCImage(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    svec=svec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    intens=intens,
                    point3D_ids=point3D_ids,
                    scale=scale,
                    bias=bias,
                )
    return images


def write_spcimages_text(images, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    # TODO: Add Point3D IDs to Image.
    # if len(images) == 0:
    #     mean_observations = 0
    # else:
    #     mean_observations = sum((len(img.point3D_ids) for img in images.values())) / len(images)
    HEADER = (
        "# Image list with two lines of data per image:\n"
        + "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, SX, SY, SZ, CAMERA_ID, NAME, SCALE, BIAS\n"
        + "#   POINTS2D[] as (X, Y, INTENSITY, POINT3D_ID)\n"
        + "# Number of images: {}, mean observations per image: {}\n".format(len(images), "TBD")
    )

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, img in images.items():
            if img.intens is None:
                continue
            image_header = [img.id, *img.qvec, *img.tvec, *img.svec, img.camera_id, img.name, img.scale, img.bias]
            first_line = " ".join(map(str, image_header))
            fid.write(first_line + "\n")

            points_strings = []
            # for xy, inten, point3D_id in zip(img.xys, img.intens, img.point3D_ids):
            for xy, inten, p3d_id in zip(img.xys, img.intens, img.point3D_ids):
                points_strings.append(" ".join(map(str, [*xy, inten, p3d_id])))
            fid.write(" ".join(points_strings) + "\n")


def read_spcpoints3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                nvec = np.array(tuple(map(float, elems[4:7])))
                albedo = float(elems[7])
                rgb = np.array(tuple(map(int, elems[8:11])))
                error = float(elems[11])
                image_ids = np.array(tuple(map(int, elems[12::2])))
                point2D_idxs = np.array(tuple(map(int, elems[13::2])))
                points3D[point3D_id] = SPCPoint3D(
                    id=point3D_id,
                    xyz=xyz,
                    nvec=nvec,
                    albedo=albedo,
                    rgb=rgb,
                    error=error,
                    image_ids=image_ids,
                    point2D_idxs=point2D_idxs,
                )
    return points3D


def write_spcpoints3D_text(points3D, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    if len(points3D) == 0:
        mean_track_length = 0
    else:
        mean_track_length = sum((len(pt.image_ids) for _, pt in points3D.items())) / len(points3D)
    HEADER = (
        "# 3D point list with one line of data per point:\n"
        + "#   POINT3D_ID, X, Y, Z, NX, NY, NZ, ALBEDO, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        + "# Number of points: {}, mean track length: {}\n".format(len(points3D), mean_track_length)
    )

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, pt in points3D.items():
            if pt.albedo is None:
                continue
            point_header = [pt.id, *pt.xyz, *pt.nvec, pt.albedo, *pt.rgb.astype(int), pt.error]
            fid.write(" ".join(map(str, point_header)) + " ")
            track_strings = []
            for image_id, point2D in zip(pt.image_ids, pt.point2D_idxs):
                track_strings.append(" ".join(map(str, [image_id, point2D])))
            fid.write(" ".join(track_strings) + "\n")


def read_spc_model(path, ext=""):
    # try to detect the extension automatically
    if ext == "":
        if detect_model_format(path, ".bin"):
            ext = ".bin"
        elif detect_model_format(path, ".txt"):
            ext = ".txt"
        else:
            print("Provide model format: '.bin' or '.txt'")
            return

    if ext == ".txt":
        cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        images = read_spcimages_text(os.path.join(path, "images" + ext))
        points3D = read_spcpoints3D_text(os.path.join(path, "points3D") + ext)
    else:
        raise RuntimeError("Support for binary files not supported.")
    return cameras, images, points3D


def write_spc_model(cameras, images, points3D, path, ext=".txt"):
    if ext == ".txt":
        write_cameras_text(cameras, os.path.join(path, "cameras" + ext))
        write_spcimages_text(images, os.path.join(path, "images" + ext))
        write_spcpoints3D_text(points3D, os.path.join(path, "points3D") + ext)
    else:
        raise RuntimeError("Support for binary files not supported.")
    return cameras, images, points3D

## Read in final reconstruction
results_path = "outputs/phomo_mod_cornelia/dn-splatter/2025-07-24_101326/meshcd"
cameras, images, points3d = read_spc_model(results_path)

## Visualize scene
import visu3d as v3d
import numpy as np


# Generate visu3d inputs.
all_albedos_list = np.array([_p3d.albedo for _p3d in points3d.values() if _p3d.albedo])
max_albedo, min_albedo = np.max(all_albedos_list), np.min(all_albedos_list)
norm_factor = max_albedo - min_albedo
rgb = np.array([int((_p3d.albedo - min_albedo) / norm_factor * 255) * np.ones(3).astype(int) for _p3d in points3d.values()])
pcd = np.array([_p3d.xyz for _p3d in points3d.values() if _p3d.albedo])
nvecs = np.array([_p3d.nvec for _p3d in points3d.values() if _p3d.albedo])

# Format cameras.
cams = []
for _i in images.values():
    _c = cameras[_i.camera_id]
    R = _i.qvec2rotmat()
    t = (-R.T @ _i.tvec.reshape((3, 1))).flatten()
    cams.append(
        v3d.Camera(
             spec=v3d.PinholeCamera.from_focal(
                resolution=(_c.height, _c.width), 
                focal_in_px=_c.params[0]
            ),
             world_from_cam=v3d.Transform(R=R.T, t=t),
        )
    )

# Plot scene!
normals_final = v3d.Ray(pos=pcd, dir=nvecs*10)
point_cloud_final = v3d.Point3d(p=pcd, rgb=rgb)
v3d.fig_config.num_samples_point3d = 50000
v3d.fig_config.num_samples_ray = 20000
v3d.fig_config.cam_scale = 100
v3d.make_fig([point_cloud_final])  # point cloud only
# v3d.make_fig([point_cloud_final, *cams])  # point clound + cameras
# v3d.make_fig([point_cloud_final, normals_final])  # point cloud + normals

## Plot albedo and normal maps
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize


plt.rcParams['figure.dpi'] = 150

# Get albedos and scale.
pcd = np.array([p3d.xyz for p3d in points3d.values()])
alb = np.array([p3d.albedo for p3d in points3d.values()])

# Project into first image. 
R_CA = images[0].qvec2rotmat()
r_AC_C = images[0].tvec[..., None]
fx, fy, cx, cy = cameras[0].params
xyz_C = R_CA @ pcd.T + r_AC_C
x_C = fx * xyz_C[0] / xyz_C[2] + cx
y_C = fy * xyz_C[1] / xyz_C[2] + cy
xy_C = np.vstack((x_C[None, ...], y_C[None, ...]))

# Plot albedo map!
norm = Normalize(np.nanmin(alb), np.nanmax(alb))
sc = plt.scatter(xy_C[0], xy_C[1], s=0.15, c=alb, marker=",", cmap="gray", norm=norm)
cbar = plt.colorbar(sc, orientation="vertical", shrink=0.87, pad=0.0)
cbar.set_label(label=r"Albedo", size=14)
plt.gca().invert_yaxis()
plt.gca().set_aspect("equal")
plt.axis("off")
plt.show()

# Plot normal map
vis_normal = lambda normal: ((normal - np.nanmin(normal[:])) / 2)[..., ::-1]

nvecs = np.array([p3d.nvec for p3d in points3d.values()])

plt.close()
plt.scatter(xy_C[0], xy_C[1], s=0.15, c=vis_normal(nvecs), marker=",")
plt.axis("off")
plt.gca().invert_yaxis()
plt.gca().set_aspect("equal")
plt.show()

## Compare image and rendering
from utils.reflectance import schroder2013_reflectance


image_id = 0

# Compute estimated brightness.
image = images[image_id]
intens_est = np.ones_like(image.intens) * np.nan
for idx, point3d_id in enumerate(image.point3D_ids):
    if point3d_id < 0:
        continue
    p3d = points3d[point3d_id]
    reflect, alpha, beta, phi = schroder2013_reflectance(image.qvec2rotmat(), image.tvec, p3d.xyz, image.svec, p3d.nvec)
    intens_est[idx] = image.scale * p3d.albedo * reflect + image.bias

# Plot measured example.
valid_idxs = ~np.isnan(intens_est)
xys = images[image_id].xys[valid_idxs]
inten = images[image_id].intens[valid_idxs]

plt.subplot(1, 2, 1)
plt.scatter(xys[:, 0], -xys[:, 1] + 1024, s=0.05, c=inten, marker=",", cmap="gray")
plt.gca().set_aspect("equal")
plt.gca().set_title("Image")
plt.axis("off")

# Get xys from forward projected landmarks.
valid_idxs = ~np.isnan(intens_est)
pcd = np.array([points3d[point3d_id].xyz if point3d_id >= 0 and valid_idxs[idx] else np.ones(3) * np.nan for idx, point3d_id in enumerate(images[image_id].point3D_ids)])
R_CA = images[image_id].qvec2rotmat()
r_AC_C = images[image_id].tvec[..., None]

fx, fy, cx, cy = cameras[0].params
xyz_C = R_CA @ pcd.T + r_AC_C
x_C = fx * xyz_C[0] / xyz_C[2] + cx
y_C = fy * xyz_C[1] / xyz_C[2] + cy
xys = np.vstack((x_C[None, ...], y_C[None, ...]))

# Plot estimated example.
plt.subplot(1, 2, 2)
plt.scatter(xys[0], -xys[1] + 1024, s=0.05, c=intens_est, marker=",", cmap="gray")
plt.gca().set_aspect("equal")
plt.gca().set_title("Rendering")
plt.axis("off")
plt.show()

print(
    "Mean photometric error:", 
    np.sqrt(np.mean((inten - intens_est[valid_idxs]) * (inten - intens_est[valid_idxs]))) / np.mean(inten),
)
