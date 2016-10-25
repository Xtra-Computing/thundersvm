mascot svmcv
======
MASCOT is a GPU-based implementation for SVM cross-validation. It also has modules for SVM classification. We will upgrade the software to support SVM regression soon.

The associated paper of this source code is: "MASCOT: Fast and Highly Scalable SVM Cross-validation using GPUs and SSDs" published in ICDM 2014.

Report bugs to: name@domain where name=wenzeyi and domain=(google's email)

This software is licensed under Apache Software License v2.0.

---------
Requirement(s):
CUDA 6.0 or later

---------
FAQ:

1. How can I use the source code?<br>
<b>A</b>: Please download the repository and then issue "make" command under the fold where our Makefile is located. After the command is completed, you will see a binary file named "mascot" in the "bin" folder. To start playing it, try command "./mascot -g 0.382 -c 100 -f 200 gisette_scale". The gisette_scale dataset is available <a href="http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#gisette">here</a> in LibSVM site.

2. What is the format of the input file?<br>
<b>A</b>: The file format is the same as the format of files in LibSVM site.

3. Does this version support the Windows OS?<br>
<b>A</b>: No. However, the code should work on Windows OS, although we have tested it on Windows.

4. Do I have to install an SSD if I want to use MASCOT?<br>
<b>A</b>: No. MASCOT works fine with HDDs, although SSDs would help improve the efficiency.

5. What are the meanings of the options?<br>
<b>A</b>: -g is for setting the gamma value; -c is for setting the penalty value of C; -f is to let MASCOT know the data dimensionality (this parameter may be removed in latter version).

6. I got "error while loading shared libraries: libcudart.so.6.0: wrong ELF class: ELFCLASS32", when I run the executable file "mascot".<br>
<b>A</b>: Running the command ''sudo ldconfig /usr/local/cuda/lib64'' should resolve the problem..
