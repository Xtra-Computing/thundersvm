Frequently Asked Questions (FAQs)
======
This page is dedicated to summarizing some frequently asked questions about ThunderSVM.

## FAQs of users
* **How can I use the source code?** 
   Please refere to [How To](how-to.md) page.

* **What is the data format of the input file?** 
  ThunderSVM uses the LibSVM format.

* **Can ThunderSVM run on CPUs?** 
  Yes. Please see [Working without GPUs](get-started.md).
  
 * **How can I do grid search?**
 Please refer to [How To](how-to.md) page.

## FAQs of developers
* **Why not use shrinking?**
  Shrinking is used in ThunderSVM and is implemented by the working set size. We don't provide the shrinking option to users, because the traditional way of shrinking is inefficient on GPUs.
