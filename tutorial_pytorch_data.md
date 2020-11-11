# Beautiful (ML) Data: Patterns&Best Practice for effective Data solutions with PyTorch


### Abstract

Data is essential in Machine Learning, and PyTorch offers a very Pythonic solution to load complex and heterogeneous dataset. However, data loading is merely the first step: preprocessing|batching|sampling|partitioning|augmenting. This tutorial explores the internals of torch.utils.data, and describes patterns and best practices for elegant data solutions in Machine learning with PyTorch.



### Description

Data processing is at the heart of every Machine Learning (ML) model training&evaluation loop; and PyTorch has revolutionised the way in which data is managed. Very Pythonic `Dataset` and `DataLoader` classes substitutes substitutes (_nested_) `list` of Numpy `ndarray`. However data `loading` is merely the first step. Data `preprocessing|sampling|batching|partitioning` are fundamental operations that are usually required in a complete ML pipeline. 

If not properly managed, this could ultimately lead to lots of _boilerplate_ code,  _re-inventing the wheel_ ™. 
This tutorial will dig into the internals of `torch.utils.data` to present patterns and best practice to load heterogeneous and custom dataset in the most  elegant and Pythonic way. 

The tutorial is organised in four parts, each focusing on specific patterns for ML data and scenarios. These parts will share the same internal structure: (I) _general introduction_; (II) _case study_. The first section will provide a technical introduction of the problem, and a description of the `torch` internals. Case studies are then used to deliver concrete examples, and application, as well as engaging with the audience, and fostering the discussion.  _Off-the-shelf_ and/or _custom_ heterogeneuous datasets will be used to comply with the broadest possible interests from the audience (e.g. Images, Text, Mixed-Type Datasets).

#### Outline

1. Intro to `Dataset`  and `DataLoader`
	* `torch.utils.data.Dataset` at a glance
	* Type of Dataset: `IterableDataset` and _Map-Style_ Dataset
	* _Case study_: File-base _vs_ Database Dataset
		* Streaming data from MongoDB
		* Dataset Composition:  `Concat`, `Chain`, `__add__`
2. Data PreProcessing and Transformation
	* `torchvision` transformers 
	* _Case Study_: Custom transformers
		* Transformer pipelines with `torchvision.transforms.Compose`
3. Data Partitioning (training / validation / test ): the PyTorch way
	* One _Dataset_ is One `Dataset`
	* Subset and `random_split`
	* _Case Study_: `Dataset` and Cross-Validation
		* How to combine `torch.utils.data.Dataset` and `sklearn.model_selection.KFold` (without using `skorch`)
		* Combining Data Partitioning and Transformers
4. Data Loading and Sampling
	* `torch.utils.data.DataLoader` and _data batching_
	* Single- and Multi-processing Data Loading
	* Data sampling: `SequentialSampling`, `RandomSumpling`
	* _Case Study_: Cross Validation Partitioning _Reviewed_: 
		* Subset & Sampling with `SequentialSubsetSampling` and `RandomSubsetSampling`

##### Pre-requisites

Basic concepts of Machine/Deep learning Data processing are required to attend this tutorial. Similarly, proficiency with the Python language and the Python Object Model is also required. Basic knowledge of the PyTorch main features is preferable.