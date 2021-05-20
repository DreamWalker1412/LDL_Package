***************************************************************************
MATLAB toolbox for Label Distribution Learning(LDL)
%Version 1.3.0      1st-May-2016

Copyright
Xin Geng (xgeng@seu.edu.cn)
School of Computer Science and Engineering, Southeast University
Nanjing 211189, P.R.China
***************************************************************************

0. Contents
===========================================================================
0. Contents
1. Introduction
2. Requirements
3. Installation
4. How to start?

1.Introduction
===========================================================================
This package implements a novel machine learning paradigm named Label Distribution Learning (LDL). A label distribution covers a certain number of labels, representingthe degree to which each label describes the instance. LDL is a general learning framework which includes both single-label and multi-label learning as its special cases. Further details about LDL can be found in the following papers:
[1] X. Geng. Label Distribution Learning. IEEE Transactions on Knowledge and Data Engineering (IEEE TKDE), 2016, in press.
[2] X. Geng, C. Yin, and Z.-H. Zhou. Facial Age Estimation by Learning from Label Distributions. IEEE Transactions on Pattern Analysis and Machine Intelligence (IEEE TPAMI), 2013, 35(10): 2401-2412.

This package can be used freely for academic, non-profit purposes. 
If you intend to use it for commercial development, please contact us. 
In academic papers using this package, 
the following references will be appreciated:
[1] X. Geng. Label Distribution Learning. IEEE Transactions on Knowledge and Data Engineering (IEEE TKDE), 2016, in press.
[2] X. Geng, C. Yin, and Z.-H. Zhou. Facial Age Estimation by Learning from Label Distributions. IEEE Transactions on Pattern Analysis and Machine Intelligence (IEEE TPAMI), 2013, 35(10): 2401-2412.

This package can be downloaded from 
http://cse.seu.edu.cn/PersonalPage/xgeng/LDL.htm. 
Please feel free to contact us if you find anything wrong or you have any further questions.

2. Requirements
===========================================================================
- Matlab, version 2013a and higher.
- The package is mostly self-contain. 
Several functions require the Optimization Toolbox. 

3. Installation
===========================================================================
- Create a directory of your choice and copy the toolbox there.
- Set the path in your Matlab to add the directory you just created.

4. How to start?
===========================================================================
We have implemented four LDL algorithms in this package, namely IIS-LLD, BFGS-LLD, CPNN, LDSVR, AA-BP, AA-kNN, PT-Bayes and PT-SVM. To help you start working with LDL, we provide eight demos (See iislldDemo.m, bfgslldDemo.m, cpnnDemo.m, ldsvrDemo.m, aabpDemo.m, ptbayesDemo.m, ptsvmDemo.m)  in this package. 
To assecss the performence, we provide fourteen measures in 'measures' fold, which contains two versions¡ª¡ªcanberra£¬chebyshev£¬cosine, squared chord, clark, inner product, pearsonX and wave hedges are the new ones, which are strongly recommended, the others like euclideandist, fidelity, intersection, kldist, sorensendist, squaredxdist are the old ones.
Before using the LDL Matlab toolbox, you'd better pre-process your datasetincluding the normalization of features and labels. When you construct the label distribution, you should ensure that the sum of distribution is equal to 1.The preprocess part depends on the specific data used. 

Please read and play with the demos to get started. Have fun!


