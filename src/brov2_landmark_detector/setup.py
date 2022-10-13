from setuptools import setup

package_name = 'brov2_landmark_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Vegard Haraldstad',
    maintainer_email='vegard.haraldstad@gmail.com',
    description='Landmark detector for detecting landmarks in 1D SSS swaths',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'brov2_landmark_detector_exe = brov2_landmark_detector.landmark_detector_main:main'
        ],
    },
)
