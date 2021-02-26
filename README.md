# Text Generator Notes

## Objective

This is an investigation into using text generation, following the a few Google Colab notebooks I found:

* [Using Hugging Face to generate text](https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/02_how_to_generate.ipynb
) which goes into some of the details behind GPT2-style text generation using improved transformer architecture and decoding methods.
* [Using BERT to generate text](https://colab.research.google.com/github/deepmipt/DeepPavlov/blob/docs/transformers-tutorial/examples/bert_generator.ipynb),
* [Training a New Language Model from Scratch](https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb)
* [Question Answering on SQUAD](https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/question_answering.ipynb)

...as well as discussions into putting this into production and any other interesting findings along the way.

## A Word on Colab

Throughout the start of this project, I attempted to use a free CoLab notebook to get things going. The advantage of CoLab is that it's well known, and free. The disadvantage is that it is very limited in terms of how it installs dependencies, it abstracts away a lot of the linux management, essentially getting in the way of appropriately installing modules.

I began by attempting to trudge my way through [CoLab with DeepPavlov](/colaberrors/deeppavloverrors.md) which created a lot of errors. I had the same issue with [CoLab with HuggingFace](/colaberrors/huggingfaceerrors.md)

So, I went and installed a Jupyter Notebook that [works with my local machine's NVidia GeForce GTX1700](https://github.com/pwdel/nvidialubuntutensorflow).

## Hugging Face Notebook

There is a [Google Colab notebook from Hugging Face](https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/02_how_to_generate.ipynb
) which goes into some of the details behind GPT2-style text generation using improved transformer architecture and decoding methods.

### Summary Notebook:

1. Notebook overview of why text generation has improved in recent years. This has included developments in transformer architecture, availability of unsupervised training, and better decoding methods. Decoding methods will be defined in this notebook.
2. Talk about what "auto-regressive" language generation means. Basically, it's the assumption that the probability distribution of a word sequence can be broken down into the product of the conditional next word distributions.
3. The fact that auto-regressive language generation is available for GPT2, XLNet, OpenAi-GPT, CTRL, TransfoXL, XLM, Bart, T5 in both PyTorch and Tensorflow >= 2.0!
4. The fact that this notebook is a tour of several decoding methods, including, "Greedy Search," "Beam Search," "Top-K Sampling" and "Top P Sampling."

### Summary of Methods:

#### Greedy Search

1. Greedy search is a tree-based structure, which selects the word with the highest probability as the next word. So for example given the word, "The" it will select either, "dog," "nice," "car," based upon simply which word has the highest probability of being next.

Feeding in the sentence, "I enjoy walking with my cute dog," the system generates text until a given output length of characters.

We are using a pre-trained GPT2 dataset. The documentation for [gpt2tokenizer is here](https://huggingface.co/transformers/model_doc/gpt2.html#gpt2tokenizer).

The documentation for [tfgpt2lmheadmodel is here](https://huggingface.co/transformers/model_doc/gpt2.html#gpt2lmheadmodel).

1. Tokenization with gpt2tokenizer, using pre-trained model. This works based upon byte-level pair encoding.
2. model using tfgpt2lmheadmodel, using pre-trained model. 



# References

[Explination of Autograd for PyTorch](http://seba1511.net/tutorials/beginner/former_torchies/autograd_tutorial.html)
