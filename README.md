mascot
======
MASCOT is a GPU-based implementation for SVM cross-validation.

The associated paper of this source code is: "MASCOT: Fast and Highly Scalable SVM Cross-validation using GPUs and SSDs" published in ICDM 2014.

You may report bugs to: name@domain, where name=zeyiw and domain=student.unimelb.edu.au

---------
FAQ:
1. How can I use the source code?
Please download the repository and then issue "make" command under the fold where our Makefile is located. After the command is completed, you will see a binary file named "mascot" in the "bin" folder. To start playing it, try command "./mascot -g 0.382 -c 100 -f 200 data/gisette_scale".

2. What is the format of the input file?
The file format is the same as the format of files in LibSVM site.

3. Does this version support the Windows OS?
No. We will release a new version that supports both Linux and Windows soon, when we finish some basic testing.

4. Do I have to install an SSD if I want to use MASCOT?
No. MASCOT works fine with HDDs, although SSDs would help improve the efficiency.

5. What are the meanings of the options?
-g is for setting the gamma value; -c is for setting the penalty value of C; -f is to let MASCOT know the data dimensionality (this parameter may be removed in latter version).
