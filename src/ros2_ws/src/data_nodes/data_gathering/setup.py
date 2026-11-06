from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'data_gathering'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='gds2011@libero.it',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'view_node = data_gathering.view_node2:main', 
            'model_data_sync_node = data_gathering.models_data_sync:main', 
            'image_input_node = data_gathering.image_gathering_realsense:main',
            'image_ari_input_node = data_gathering.image_ari_gathering:main',
        ],
    },
)
