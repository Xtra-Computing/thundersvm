from os import path
import setuptools
from shutil import copyfile
from sys import platform


dirname = path.dirname(path.abspath(__file__))

if platform == "linux" or platform == "linux2":
	lib_path = path.abspath(path.join(dirname, '../build/lib/libthundersvm.so'))
elif platform == "win32":
	lib_path = path.abspath(path.join(dirname, '../build/bin/Debug/thundersvm.dll'))
elif platform == "darwin":
	lib_path = path.abspath(path.join(dirname, '../build/lib/libthundersvm.dylib'))
else :
	raise EnvironmentError("OS not supported!")

if not path.exists(path.join(dirname, "thundersvm", path.basename(lib_path))):
	copyfile(lib_path, path.join(dirname, "thundersvm", path.basename(lib_path)))
setuptools.setup(name="thundersvm",
			     version="0.0.2",
				 packages=["thundersvm"],
				 package_dir={"python": "thundersvm"},
			     description="A Fast SVM Library on GPUs and CPUs",
			     long_description="The mission of ThunderSVM is to help users easily and efficiently apply SVMs to solve problems. ThunderSVM exploits GPUs and multi-core CPUs to achieve high efficiency",
			     long_description_content_type="text/plain",
			     url="https://github.com/zeyiwen/thundersvm",
			     package_data = {"thundersvm": [path.basename(lib_path)]},
			     install_requires=['numpy','scipy','scikit-learn'],
			     classifiers=[
			                  "Programming Language :: Python :: 3",
			                  "License :: OSI Approved :: Apache Software License",
			    ],
				 python_requires=">=3"
)
