This folder contains all data and code to run relevant analyses and reproduce the figures 
in "Revealing the multidimensional mental representations of natural objects underlying 
human similarity judgments" by M.N. Hebart, C.Y. Zheng, F. Pereira, and C.I. Baker.
Nature Human Behaviour 4, 1173-1185. http://rdcu.be/b8pqd 
Preprint: https://psyarxiv.com/7wrgh

To run all relevant analyses and reproduce the figures, open Matlab R2016b or later 
(earlier versions were not tested but might work), navigate to the unzipped folder 
containing all files, and run make_figures_behavsim. For running only parts, jump 
directly into the script. The total execution time should be 10-20 minutes the first time, 
and 1-2 minutes the second time.

Edit 2020/11/21: Minor bug fixed in make_figures_behavsim and embedding2sim that affects 
results with similarity matrices, leading to changes in the order of ~0.001. All results 
were checked and were unchanged given the numerical precision of the presented results.