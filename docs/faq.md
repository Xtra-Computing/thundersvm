Frequently Asked Questions (FAQs)
======
This page is dedicated to summarizing some frequently asked questions about ThunderSVM.

## FAQs of users
* How can I use the source code?  
  Please refere to [How To](how-to.md) page.

* What is the format of the input file?  
  The file format is the same as the format of files in LibSVM site.

* Does this version support the Windows OS?  
  No. However, the code should work on Windows OS, although we haven't tested it on Windows.

* I got "error while loading shared libraries: libcudart.so.6.0: wrong ELF class: ELFCLASS32", when I run the executable file "thundersvm".  
  Running the command ```sudo ldconfig /usr/local/cuda/lib64``` should resolve the problem..

* Can ThunderSVM run on CPUs?  
  No. The current version of ThunderSVM must run on GPUs, but we strive to make ThunderSVM run purely on CPUs in parallel.
  
  * How can I do grid search?
  Since ThunderSVM supports cross-validation. You can write a simple grid.sh script like the following one
```
#!/usr/bin/bash
DATASET=$1
OPTIONS=
N_FOLD=5
for c in 1 3 10 30 100
do
  for g in 0.1 0.3 1 3 10
    do
      bin/thundersvm-train -c ${c} -g ${g} -v ${N_FOLD} ${OPTIONS} ${DATASET}
    done
done
```
Then ```run sh grid.sh [dataset]```.  You may modify the script to meet your needs. Indeed, ThunderSVM supports the same command line parameters as LIBSVM. So the script grid.py in LIBSVM can be used for ThunderSVM with minor modifications.

## FAQs of developers
* Why not use shrinking?  
  Shrinking is used in ThunderSVM, and is implemented by the working set size. We don't provide the shrinking option to users, because the traditional way of shrinking is inefficient on GPUs.

