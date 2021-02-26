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

1. Tokenization with gpt2tokenizer, using pre-trained model. This works based upon byte-level pair encoding. This was based upon a model built using 40GB of text data.
2. model using tfgpt2lmheadmodel, using pre-trained model. This basically uses linear weights tied to the input embedding.
3. Encodes with tokenizer.encode and for the input id's.
4. model.generate to generate the model with a specified maximum length.

#### Beam search

Beam search reduces the risk of missing hidden high probability word sequences by keeping the most likely num_beams of hypotheses at each time step and eventually choosing the hypothesis that has the overall highest probability.

1. beam_output after generating a model with a specified number of beams and maximum length.

This output still contains repetitions of the same sequences.

2. Introduce n-grams, a sequence of n words penalties. This ensures that no n-gram appears twice by manually setting the probability of next words that could create an already seen n-gram to 0.

When doing this, we see that repetition gets reduced. Here was our input and output:

```
beam_output = model.generate(
    input_ids,
    max_length=100,
    num_beams=5,
    no_repeat_ngram_size=4,
    early_stopping=True
)
```
Output
```
We have a lot of new SOCs in stock, and we are working hard to get them into the hands of our customers as quickly as possible."

The company said it was working closely with the U.S. Department of Homeland Security and the Federal Bureau of Investigation.

"We will continue to work closely with our partners to ensure that our customers have the best experience with our products and services," the company said in a statement. "We are committed to providing our customers with the...
```
Interestingly, the output starts to use language from what appears to be some kind of extremely negative event!

There are some notes about Beam search, mentioning that it works well with tasks where the length of the desired generated output is predictable. However it doesn't work as well for open-ended generation, such as dialog and story generation.

Beam search suffers heavily from repetitive generation. This is especially hard to control with n-gram or other penalties in story generation.

High quality human language does not follow a distribution of high probability next words. In other words, we as humans want generated text to surprise us and no be boring or predictable. BeamSearch in general is, "not surprising," because it is be default, being "safe" with how it predicts words.

####


# References

[Explination of Autograd for PyTorch](http://seba1511.net/tutorials/beginner/former_torchies/autograd_tutorial.html)
