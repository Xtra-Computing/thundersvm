Frequently Asked Questions (FAQs)
======
This page is dedicated to summarizing some frequently asked questions about ThunderSVM.

## FAQs of users
* **How can I use the source code?**  
   Please refere to [How To](how-to.md) page.

* **What is the format of the input file?**  
  The file format is the same as the format of files in LibSVM site.

* **Does this version support the Windows OS?**  
  No. However, the code should work on Windows OS, although we haven't tested it on Windows.

* **Can ThunderSVM run on CPUs?**  
  Yes. Please see [Working without GPUs](get-started.md#working-without-gpus).
  
 * **How can I do grid search?**
   Since ThunderSVM supports cross-validation. You can write a simple grid.sh script like the following one. Then ```run sh grid.sh [dataset]```.  You may modify the script to meet your needs. Indeed, ThunderSVM supports the same command line parameters as LIBSVM. So the script grid.py in LIBSVM can be used for ThunderSVM with minor modifications.
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

## FAQs of developers
* **Why not use shrinking?  **
  Shrinking is used in ThunderSVM, and is implemented by the working set size. We don't provide the shrinking option to users, because the traditional way of shrinking is inefficient on GPUs.

