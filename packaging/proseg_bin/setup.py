from wheel.bdist_wheel import bdist_wheel
from setuptools import setup


class LinuxBinaryWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        return "py3", "none", "manylinux_2_34_x86_64"


setup(cmdclass={"bdist_wheel": LinuxBinaryWheel})
