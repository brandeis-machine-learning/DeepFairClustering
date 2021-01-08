# Deep Fair Clustering

Peizhao Li, Han Zhao, and Hongfu Liu. "[Deep Fair Clustering for Visual Learning](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Deep_Fair_Clustering_for_Visual_Learning_CVPR_2020_paper.html)", CVPR 2020.

Fair clustering aims to hide sensitive attributes during
data partition by balancing the distribution of protected
subgroups in each cluster. Existing work attempts to address this problem by reducing it to a classical balanced
clustering with a constraint on the proportion of protected
subgroups of the input space. However, the input space
may limit the clustering performance, and so far only lowdimensional datasets have been considered. In light of these
limitations, in this paper, we propose Deep Fair Clustering
(DFC) to learn fair and clustering-favorable representations for clustering simultaneously. Our approach could effectively filter out sensitive attributes from representations,
and also lead to representations that are amenable for the
following cluster analysis. Theoretically, we show that our
fairness constraint in DFC will not incur much loss in terms
of several clustering metrics. Empirically, we provide extensive experimental demonstrations on four visual datasets
to corroborate the superior performance of the proposed
approach over existing fair clustering and deep clustering
methods on both cluster validity and fairness criterion.

## Experiment on MNIST-USPS dataset

Using Pytorch 1.7

1. Put image data following `./data/train_mnist.txt` and `./data/train_usps.txt`
2. Run `python main.py`

## Reference
    @InProceedings{Li_2020_CVPR,
    author = {Li, Peizhao and Zhao, Han and Liu, Hongfu},
    title = {Deep Fair Clustering for Visual Learning},
    booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
    }