FAQ:
======
* How can I use the source code?
Please refere to [How To](how-to.md) page.

* What is the format of the input file?
The file format is the same as the format of files in LibSVM site.

* Does this version support the Windows OS?
No. However, the code should work on Windows OS, although we haven't tested it on Windows.

* I got "error while loading shared libraries: libcudart.so.6.0: wrong ELF class: ELFCLASS32", when I run the executable file "thundersvm".
Running the command ```sudo ldconfig /usr/local/cuda/lib64``` should resolve the problem..

