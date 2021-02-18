# Text Generator Notes

## Objective

This is an investigation into using text generation, following the [Google Colab Notebook that I found which uses BERT to generate text](https://colab.research.google.com/github/deepmipt/DeepPavlov/blob/docs/transformers-tutorial/examples/bert_generator.ipynb), as well as discussions into putting this into production and any other interesting findings along the way.

## Exploring the Colab Notebook

The notebook was out of date, so one of the first steps out-of-the box was to update and investigate the various Python modules.

### deeppavlov, torch and transformers

The first thing that the text generator notebook has us do is install deeppavlov, [torch](https://pypi.org/project/torch/) and transformers.

```
!pip install deeppavlov==0.8.0 torch==1.4.0 transformers==2.8.0
```

* [DeepPavlov](https://demo.deeppavlov.ai/#/en/textqa), or [deeppavlov at python](https://pypi.org/project/deeppavlov/) is deisgned to be a neural network powered conversational AI engine. Interestingly, [Deepavlov has its own docker registry](https://hub.docker.com/u/deeppavlov). The majority of [DeepPavlov](https://github.com/deepmipt/DeepPavlov/graphs/contributors) seems to have been written around 2018.
* [PyTorch](https://pypi.org/project/torch/) is a Python package that provides two main high-level features: Tensor Computation, like NumPy, with a strong GPU accelleration, and Deep Neural Networks built on a tape-based autograd system. This means that all of the operations that it undertook, it will remember on the, "tape" and it will replay the operations. [PyTorch is well maintained and has been picking up in volume over the years since 2017](https://github.com/pytorch/pytorch/graphs/contributors).
* [Transformers](https://pypi.org/project/transformers/) was started [recently in 2019](https://github.com/huggingface/transformers/graphs/contributors) as a large system of pretrained models for various tasks that can be performed on texts such as classification, information extraction, question answering summarization, translation, text generation and more.


## A Word on RAM

Upgrading CoLab RAM for free

https://towardsdatascience.com/upgrade-your-memory-on-google-colab-for-free-1b8b18e8791d

## Spinning Up My Own Text Generation

https://stackoverflow.com/questions/60142937/huggingface-transformers-for-text-generation-with-ctrl-with-google-colabs-free

How to generate text:

https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/02_how_to_generate.ipynb


How to train new language model from scratch

https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb

Question Answering on SQUAD

https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/question_answering.ipynb

# References

[Explination of Autograd for PyTorch](http://seba1511.net/tutorials/beginner/former_torchies/autograd_tutorial.html)