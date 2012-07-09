#! /usr/bin/python

import numpy as np

import roslib
roslib.load_manifest('hrl_ellipsoidal_control')

import tf.transformations as tf_trans

from hrl_generic_arms.pose_converter import PoseConverter

class EllipsoidSpace(object):
    def __init__(self, E=1, is_oblate=False):
        self.A = 1
        self.E = E
        #self.B = np.sqrt(1. - E**2)
        self.a = self.A * self.E
        self.is_oblate = is_oblate

    def load_ell_params(self, E, is_oblate=False, height=1):
        self.E = E
        self.a = self.A * self.E
        self.is_oblate = is_oblate
        self.height = height

    def ellipsoidal_to_cart(self, lat, lon, height):
        #assert height > 0 and lat >= 0 and lat <= np.pi and lon >= 0 and lon < 2 * np.pi
        if not self.is_oblate:
            x = self.a * np.sinh(height) * np.sin(lat) * np.cos(lon)
            y = self.a * np.sinh(height) * np.sin(lat) * np.sin(lon)
            z = self.a * np.cosh(height) * np.cos(lat)
        else:
            x = self.a * np.cosh(height) * np.cos(lat-np.pi/2) * np.cos(lon)
            y = self.a * np.cosh(height) * np.cos(lat-np.pi/2) * np.sin(lon)
            z = self.a * np.sinh(height) * np.sin(lat-np.pi/2)
        pos_local = np.mat([x, y, z]).T
        return pos_local

    def partial_height(self, lat, lon, height):
        #assert height > 0 and lat >= 0 and lat <= np.pi and lon >= 0 and lon < 2 * np.pi
        if not self.is_oblate:
            x = self.a * np.cosh(height) * np.sin(lat) * np.cos(lon)
            y = self.a * np.cosh(height) * np.sin(lat) * np.sin(lon)
            z = self.a * np.sinh(height) * np.cos(lat)
        else:
            x = self.a * np.sinh(height) * np.sin(lat-np.pi/2) * np.cos(lon)
            y = self.a * np.sinh(height) * np.sin(lat-np.pi/2) * np.sin(lon)
            z = self.a * np.cosh(height) * np.cos(lat-np.pi/2)
        return np.mat([x, y, z]).T

    #def partial_v(self, lat, lon, height):
    #    #assert height > 0 and lat >= 0 and lat <= np.pi and lon >= 0 and lon < 2 * np.pi
    #    x = self.a * np.sinh(height) * np.cos(lat) * np.cos(lon)
    #    y = self.a * np.sinh(height) * np.cos(lat) * np.sin(lon)
    #    z = self.a * np.cosh(height) * -np.sin(lat)
    #    return np.mat([x, y, z]).T
    #def partial_p(self, lat, lon, height):
    #    #assert height > 0 and lat >= 0 and lat <= np.pi and lon >= 0 and lon < 2 * np.pi
    #    x = self.a * np.sinh(height) * np.sin(lat) * -np.sin(lon)
    #    y = self.a * np.sinh(height) * np.sin(lat) * np.cos(lon)
    #    z = 0.
    #    return np.mat([x, y, z]).T

    def ellipsoidal_to_pose(self, lat, lon, height):
        if not self.is_oblate:
            return self._ellipsoidal_to_pose_prolate(lat, lon, height)
        else:
            return self._ellipsoidal_to_pose_oblate(lat, lon, height)

    def _ellipsoidal_to_pose_prolate(self, lat, lon, height):
        pos = self.ellipsoidal_to_cart(lat, lon, height)
        df_du = self.partial_height(lat, lon, height)
        nx, ny, nz = df_du.T.A[0] / np.linalg.norm(df_du)
        j = np.sqrt(1./(1.+ny*ny/(nz*nz)))
        k = -ny*j/nz
        norm_rot = np.mat([[-nx,  ny*k - nz*j,  0],      
                           [-ny,  -nx*k,        j],      
                           [-nz,  nx*j,         k]])
        _, norm_quat = PoseConverter.to_pos_quat(np.mat([0, 0, 0]).T, norm_rot)
        rot_angle = np.arctan(-norm_rot[2,1] / norm_rot[2,2])
        #print norm_rot
        quat_ortho_rot = tf_trans.quaternion_from_euler(rot_angle + np.pi, 0.0, 0.0)
        norm_quat_ortho = tf_trans.quaternion_multiply(norm_quat, quat_ortho_rot)
        norm_rot_ortho = np.mat(tf_trans.quaternion_matrix(norm_quat_ortho)[:3,:3])
        if norm_rot_ortho[2, 2] > 0:
            flip_axis_ang = 0
        else:
            flip_axis_ang = np.pi
        quat_flip = tf_trans.quaternion_from_euler(flip_axis_ang, 0.0, 0.0)
        norm_quat_ortho_flipped = tf_trans.quaternion_multiply(norm_quat_ortho, quat_flip)

        pose = PoseConverter.to_pos_quat(pos, norm_quat_ortho_flipped)
        #print ("ellipsoidal_to_pose: latlonheight: %f, %f, %f" %
        #       (lat, lon, height) +
        #       str(PoseConverter.to_homo_mat(pose)))
        return pose

    def _ellipsoidal_to_pose_oblate(self, lat, lon, height):
        pos = self.ellipsoidal_to_cart(lat, lon, height)
        df_du = self.partial_height(-lat, lon, height)
        nx, ny, nz = df_du.T.A[0] / np.linalg.norm(df_du)
        j = np.sqrt(1./(1.+ny*ny/(nz*nz)))
        k = -ny*j/nz
        norm_rot = np.mat([[-nx,  ny*k - nz*j,  0],      
                           [-ny,  -nx*k,        j],      
                           [-nz,  nx*j,         k]])
        _, norm_quat = PoseConverter.to_pos_quat(np.mat([0, 0, 0]).T, norm_rot)
        rot_angle = np.arctan(-norm_rot[2,1] / norm_rot[2,2])
        #print norm_rot
        quat_ortho_rot = tf_trans.quaternion_from_euler(rot_angle, 0.0, 0.0)
        norm_quat_ortho = tf_trans.quaternion_multiply(norm_quat, quat_ortho_rot)
        quat_ortho_rot2 = tf_trans.quaternion_from_euler(0.0, np.pi/2, 0.0)
        norm_quat_ortho = tf_trans.quaternion_multiply(norm_quat_ortho, quat_ortho_rot2)
        if lon >= np.pi:
            quat_flip = tf_trans.quaternion_from_euler(0.0, 0.0, np.pi)
            norm_quat_ortho = tf_trans.quaternion_multiply(norm_quat_ortho, quat_flip)

        pose = PoseConverter.to_pos_quat(pos, norm_quat_ortho)
        #print ("ellipsoidal_to_pose: latlonheight: %f, %f, %f" %
        #       (lat, lon, height) +
        #       str(PoseConverter.to_homo_mat(pose)))
        return pose

    def pose_to_ellipsoidal(self, pose):
        pose_pos, pose_rot = PoseConverter.to_pos_rot(pose)
        lat, lon, height = self.pos_to_ellipsoidal(pose_pos[0,0], pose_pos[1,0], pose_pos[2,0])
        _, ell_rot = PoseConverter.to_pos_rot(self.ellipsoidal_to_pose(lat, lon, height))
        _, quat_rot = PoseConverter.to_pos_quat(np.mat([0]*3).T, ell_rot.T * pose_rot)
        return [lat, lon, height], quat_rot

    def pos_to_ellipsoidal(self, x, y, z):
        if not self.is_oblate:
            return self._pos_to_ellipsoidal_prolate(x, y, z)
        else:
            return self._pos_to_ellipsoidal_oblate(x, y, z)

    def _pos_to_ellipsoidal_prolate(self, x, y, z):
        lon = np.arctan2(y, x)
        if lon < 0.:
            lon += 2 * np.pi
        p = np.sqrt(x**2 + y**2)
        a = self.a
        inner = np.sqrt((np.sqrt((z**2 - a**2 + p**2)**2 + (2. * a * p)**2) / a**2 -
                         (z / a)**2 - (p / a)**2 + 1) / 2.)
        assert inner < 1.0001
        if inner > 0.9999:
            lat = np.pi/2.
        else:
            lat = np.arcsin(inner)
        if z < 0.:
            lat = np.pi - np.fabs(lat)
        if np.fabs(np.sin(lat)) > 0.05:
            if np.fabs(np.cos(lon)) > 0.05:
                use_case = 'x'
                sinh_height = x / (a * np.sin(lat) * np.cos(lon))
                height = np.log(sinh_height + np.sqrt(sinh_height**2 + 1))
            else:
                use_case = 'y'
                sinh_height = y / (a * np.sin(lat) * np.sin(lon))
                height = np.log(sinh_height + np.sqrt(sinh_height**2 + 1))
        else:
            use_case = 'z'
            cosh_height = z / (a * np.cos(lat))
            assert np.fabs(cosh_height) >= 1, ("cosh_height %f, a %f, x %f, y %f, z %f, lat %f, lon %f" %
                                               (cosh_height, a, x, y, z, lat, lon))
            height = np.log(cosh_height + np.sqrt(cosh_height**2 - 1))
        print ("%s pos_to_ellipsoidal: xyz: %f, %f, %f; latlonheight: %f, %f, %f" %
               (use_case, x, y, z, lat, lon, height))
        assert not np.any(np.isnan([lat, lon, height])), ("cosh_height %f, a %f, x %f, y %f, z %f, lat %f, lon %f" %
                                               (cosh_height, a, x, y, z, lat, lon))
        return lat, lon, height

    def _pos_to_ellipsoidal_oblate(self, x, y, z):
        lon = np.arctan2(y, x)
        if lon < 0.:
            lon += 2 * np.pi
        p = np.sqrt(x**2 + y**2)
        a = self.a
        d_1 = np.sqrt((p + a)**2 + z**2)
        d_2 = np.sqrt((p - a)**2 + z**2)
        cosh_height = (d_1 + d_2) / (2 * a)
        assert np.fabs(cosh_height) >= 1, ("cosh_height %f, a %f, x %f, y %f, z %f, lat %f, lon %f" %
                                           (cosh_height, a, x, y, z, lat, lon))
        height = np.log(cosh_height + np.sqrt(cosh_height**2 - 1))
        cos_lat = (d_1 - d_2) / (2 * a)
        lat = np.arccos(cos_lat)
        if z < 0.:
            lat *= -1

        # we're going to convert the latitude coord so it is always positive:
        lat += np.pi / 2.

        return lat, lon, height
        
def main():
    e_space = EllipsoidSpace(1)
    # test pos_to_ellipsoidal
    for xm in range(2):
        for ym in range(2):
            for zm in range(2):
                for i in range(10000):
                    x, y, z = np.random.uniform(-2.5, 2.5, 3)
                    lat, lon, height = e_space.pos_to_ellipsoidal(xm*x, ym*y, zm*z)
                    assert lat >= 0 and lat <= np.pi*1.0001, ("latlonheight: %f, %f, %f" %
                                                       (lat, lon, height))
                    assert lon >= 0 and lon < 2*np.pi*1.0001, ("latlonheight: %f, %f, %f" %
                                                       (lat, lon, height))
                    assert height >= 0, ("latlonheight: %f, %f, %f" %
                                                       (lat, lon, height))
                    #print lat, lon, height

if __name__ == "__main__":
    main()
