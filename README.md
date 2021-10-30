# Beautiful Data for Machine Learning

## Patterns & Best Practice for effective Data solutions with PyTorch

![pydata-logo](https://pydata.org/global2021/wp-content/uploads/2021/06/logo.png)
This tutorial will be presented at [PyData Global 2020](https://global.pydata.org/) conference

## Abstract

Data is essential in Machine Learning, and PyTorch offers a very Pythonic solution to load complex and heterogeneous dataset.
However, data loading is merely the first step: `preprocessing`|`batching`|`sampling`|`partitioning`|`augmenting`.

This tutorial explores the internals of `torch.utils.data`, and describes patterns and best practices for elegant data solutions
in Machine and Deep learning with PyTorch.


## Get started


If you want to start digging into examples and patterns, there is a [Cover](Cover.ipynb) notebook to get you started.


### Outline

1. Part 1 (Prelude)
    * Data Representation for Machine Learning

2. Part 2 Intro to `Dataset` and `DataLoader`
    * `torch.utils.data.Dataset` at a glance
    * Case Study: FER Dataset

3. Part 3 Data Transformation and Sampling
    * `torchvision` transformers
    * _Case Study_: Custom (Random) transformers
        * Transformer pipelines with `torchvision.transforms.Compose`
    * Data Sampling and Data Loader
        * Handling imbalanced samples in FER data

4. Part 4 Data Partitioning (training / validation / test ): the PyTorch way
    * One _Dataset_ is One `Dataset`
    * Subset and `random_split`
    * _Case Study_: `Dataset` and Cross-Validation
        * How to combine `torch.utils.data.Dataset` and `sklearn.model_selection.KFold` (without using `skorch`)
        * Combining Data Partitioning and Transformers

5. Part 5 Data Abstractions for Image Segmentation
    * `dataclass` and Python Data Model
    * Case Study for Digital Pathology
    * Working with tiles and Patches
        * Patches in Batches for Spleen Segmentation

### Description

Data processing is at the heart of every Machine Learning (ML) model _training&evaluation_ loop; and PyTorch has revolutionised the way in which data is managed.
Very Pythonic `Dataset` and `DataLoader` classes substitutes substitutes (_nested_) `list` of Numpy `ndarray`.

However data `loading` is merely the first step. Data `preprocessing`|`sampling`|`batching`|`partitioning` are fundamental operations that are usually required in a complete ML pipeline.

If not properly managed, this could ultimately lead to lots of _boilerplate_ code,  _re-inventing the wheel_ ™.
This tutorial will dig into the internals of `torch.utils.data` to present patterns and best practice to load heterogeneous and custom dataset in the most elegant and Pythonic way.

The tutorial is organised in four parts, each focusing on specific patterns for ML data and scenarios.
These parts will share the same internal structure: (I) _general introduction_; (II) _case study_.

The first section will provide a technical introduction of the problem, and a description of the `torch` internals.
Case studies are then used to deliver concrete examples, and application, as well as engaging with the audience, and fostering the discussion.
_Off-the-shelf_ and/or _custom_ heterogeneuous datasets will be used to comply with the broadest possible interests from the audience
(e.g. Images, Text, Mixed-Type Datasets).

#### Pre-requisites

Basic concepts of Machine/Deep learning Data processing are required to attend this tutorial. Similarly, proficiency with the Python language and
the Python Object Model is also required. Basic knowledge of the PyTorch main features is preferable.

#### Setting up the Python Environment

It is possible to create the Python virtual environment to run all the notebooks in this repository either using `conda` (for Anaconda Python distribution) or `pyenv` and `pip`.

To setup the Anaconda environment:
```bash
$ conda env create -f torch_beautiful_data.yml
```
This will create a new virtual environment called `torch-beautiful-data`. 

```bash
$ conda activate torch-beautiful-data
```
to activate the environment.

At this stage, you're all set and you should be ready to start playing with the notebook.
So, run a jupyter notebook server on your local computer, by running the following command in your Terminal:

```bash
$ jupyter notebook
```

Have fun! 🎉

**Note**: Alternatively, if you would prefer installing the required packages using `pip`, it is very simple. Just run the following command:

```bash
$ pip install -r requirements.txt
```

#### Acknowledgments

Public shout out to **all** [PyData Global](https://global.pydata.org/) organisers, 
and to [Matthijs](https://github.com/MBrouns) in particular for his wonderful 
support during the preparation of this tutorial!
