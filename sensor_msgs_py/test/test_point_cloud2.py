# Copyright 2020 Open Source Robotics Foundation, Inc.
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

import unittest

import numpy as np

from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


pylist = [[0.0, 0.1, 0.2],
          [1.0, 1.1, 1.2],
          [2.0, 2.1, 2.2],
          [3.0, 3.1, 3.2],
          [4.0, np.nan, 4.2]]
points = np.array(pylist, dtype=np.float32)

fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
          PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
          PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)]

itemsize = points.itemsize
data = points.tobytes()

# 3D (xyz) point cloud (nx3)
pcd = PointCloud2(
    header=Header(frame_id='frame'),
    height=1,
    width=points.shape[0],
    is_dense=False,
    is_bigendian=False,  # Not sure how to properly determine this.
    fields=fields,
    point_step=(itemsize * 3),  # Every point consists of three float32s.
    row_step=(itemsize * 3 * points.shape[0]),
    data=data
)


# 2D (yz) point cloud
fields2 = [PointField(name='y', offset=0,
                      datatype=PointField.FLOAT32, count=1),
           PointField(name='z', offset=4,
                      datatype=PointField.FLOAT32, count=1)]
pylist2 = points[:, 1:].tolist()  # y and z column.
data2 = points[:, 1:].tobytes()
pcd2 = PointCloud2(
    header=Header(frame_id='frame'),
    height=1,
    width=points.shape[0],
    is_dense=False,
    is_bigendian=False,  # Not sure how to properly determine this.
    fields=fields,
    point_step=(itemsize * 2),  # Every point consists of three float32s.
    row_step=(itemsize * 2 * points.shape[0]),
    data=data
)


class TestPointCloud2Methods(unittest.TestCase):

    def test_read_points(self):
        # Test that converting a PointCloud2 to a list, is equivalent to
        # the original list of points.
        pcd_list = list(point_cloud2.read_points(pcd))
        self.assertTrue(np.allclose(pcd_list, pylist, equal_nan=True))

    def test_read_points_field(self):
        # Test that field selection is working.
        pcd_list = list(point_cloud2.read_points(pcd, field_names=['x', 'z']))
        # Check correct shape.
        self.assertTrue(np.array(pcd_list).shape == points[:, [0, 2]].shape)
        # Check "correct" values.
        self.assertTrue(np.allclose(pcd_list, points[:, [0, 2]],
                                    equal_nan=True))

    def test_read_points_skip_nan(self):
        # Test that removing NaNs work.
        pcd_list = list(point_cloud2.read_points(pcd, skip_nans=True))
        points_nonan = points[~np.any(np.isnan(points), axis=1)]
        # Check correct shape
        self.assertTrue(np.array(pcd_list).shape == points_nonan.shape)
        # Check correct values.
        # We do not expect NaNs, so I explicitly state that NaNs aren't
        # considered equal. (This is the default behavious of `allclose()`)
        self.assertTrue(np.allclose(pcd_list, points_nonan, equal_nan=False))

    def test_read_points_list(self):
        # Check that reading a PointCloud2 message to a list is performed
        # correctly.
        points_named = point_cloud2.read_points_list(pcd)
        self.assertTrue(np.allclose(np.array(points_named), points, equal_nan=True))

    def test_create_cloud(self):
        thispcd = point_cloud2.create_cloud(Header(frame_id='frame'),
                                            fields, pylist)
        self.assertTrue(thispcd == pcd)
        thispcd = point_cloud2.create_cloud(Header(frame_id='frame2'),
                                            fields, pylist)
        self.assertFalse(thispcd == pcd)
        thispcd = point_cloud2.create_cloud(Header(frame_id='frame'),
                                            fields2, pylist2)
        self.assertFalse(thispcd == pcd)

    def test_create_cloud_xyz32(self):
        thispcd = point_cloud2.create_cloud_xyz32(Header(frame_id='frame'),
                                                  pylist)
        self.assertTrue(thispcd == pcd)


if __name__ == '__main__':
    unittest.main()
