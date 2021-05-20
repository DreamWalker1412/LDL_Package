% Label Distribution Learning Toolbox
% Version 1.3.0      1st-May-20156
%
% %
% aabpDemo          -   The example of AABP algorithm.
% aabpTrain         -   The training part of AABP algorithm.
% aabpPredict       -   To predict the distribution by trained AABP model.
% 
% aaknnDemo         -   The example of AAKNN algorithm.
% aaknn             -   The AAKNN model.
% 
% bfgslldDemo       -   The example of BFGSLLD algorithm.
% bfgslldTrain      -   The training part of BFGSLLD algorithm.
% bfgsProcess       -   Provide the target function and the gradient.
% fminlbfgs         -   Finds a local minimum of a function of several variables.
% 
% cpnnDemo          -   The example of CPNN algorithm.
% cpnn              -   The implementation of CPNN structure.
% cpnnTrain         -   The training part of CPNN algorithm.
% cpnnPredict       -   To predict the distribution by trained CPNN structure.
% 
% iislldDemo        -   The example of IISLLD algorithm.
% iislldTrain       -   The training part of IISLLD algorithm.
% lldPredict        -   To predict the distribution by trained IISLLD/BFGSLLD model.
% 
% ldsvrDemo         -   The example of LDSVR algorithm.
% ldsvrTrain        -   The training part of LDSVR algorithm.
% ldsvrmsvr         -   The LDSVR's Multioutput SVR.
% ldsvrPredict      -   To predict the distribution by trained LDSVR model.
% 
% ptbayesDemo       -   The example of PTbayes algorithm.
% ptbayesTrain      -   The training part of PTbayes algorithm.
% bayes             -   The bayes classifier of PTbayes algorithm.
% resample          -   To resample examples from  training data transformed by PTsvm/PTbayes
% ptbayesPredict    -   To predict the distribution by trained PTBayes model.
% 
% ptsvmDemo         -   The example of PTsvm algorithm.
% ptsvmTrain        -   The training part of PTsvm algorithm.
% ptsvmPredict      -   To predict the distribution by trained PTsvm model.
% 
% computeMeasures   -   Computes different measures between two distributions.
% drawDistribution  -   Draw the label distributions for comparision. 
% kernelmatrix      -   Calculate the kernel matrix of vectors (samples) between two data matrices.
% 
% % mesures file
% canberra          -   Calculate the average Canberra between the two distributions.
% chebyshev         -   Calculate the average Chebyshev between the two distributions.
% clark             -   Calculate the average Clark between the two distributions.
% cosine            -   Calculate the average Cosine between the two distributions.
% innerProduct      -   Calculate the average InnerProduct between the two distributions.
% pearsonX          -   Calculate the average PearsonX between the two distributions.
% squaredChord      -   Calculate the average Squared_Chord between the two distributions.
% waveHedges        -	Calculate the average WaveHedges between the two distributions.    
% 
% euclideandist     -   Calculate the average Euclidean distance between the predicted and the real label distribution.
% fidelity          -   Calculate the average fidelity between the two distributions.
% intersection      -   Calculate the average intersection between two distributions.
% kldist            -   Calculate the average Kullback-Leibler divergence between two distributions.
% sorensendist      -   Calculate the average Sorensen's distance between ttwo distributions.
% squaredxdist      -   Calculate the average squared x*x distance between two distributions.
% 
%% 
%   * 
%   Copyright: Xin Geng (xgeng@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%
