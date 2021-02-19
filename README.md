# Text Generator Notes

## Objective

This is an investigation into using text generation, following the [Google Colab Notebook that I found which uses BERT to generate text](https://colab.research.google.com/github/deepmipt/DeepPavlov/blob/docs/transformers-tutorial/examples/bert_generator.ipynb), as well as discussions into putting this into production and any other interesting findings along the way.

## Exploring the Colab Notebook

The notebook was out of date, so one of the first steps out-of-the box was to update and investigate the various Python modules.

## A Word on RAM

Throughout this notebook, starting off with, I'm using the default free RAM plan. However at some point, we might find that this is not enough. CoLab does offer a high RAM plan at $10/month to upgrade to 25GB of RAM vs. the standard 


### Installing deeppavlov, torch and transformers

The first thing that the text generator notebook has us do is install deeppavlov, [torch](https://pypi.org/project/torch/) and transformers.

```
!pip install deeppavlov==0.8.0 torch==1.4.0 transformers==2.8.0
```
So the first thing we did was updated the above plugins to the most recent versions so that they would be compatible.  The original versions that were listed resulted in an error when running pip.  Let's take a close look at what each module is and what it's doing for us.

* [DeepPavlov](https://demo.deeppavlov.ai/#/en/textqa), or [deeppavlov at python](https://pypi.org/project/deeppavlov/) is deisgned to be a neural network powered conversational AI engine. Interestingly, [Deepavlov has its own docker registry](https://hub.docker.com/u/deeppavlov). The majority of [DeepPavlov](https://github.com/deepmipt/DeepPavlov/graphs/contributors) seems to have been written around 2018.
* [PyTorch](https://pypi.org/project/torch/) is a Python package that provides two main high-level features: Tensor Computation, like NumPy, with a strong GPU accelleration, and Deep Neural Networks built on a tape-based autograd system. This means that all of the operations that it undertook, it will remember on the, "tape" and it will replay the operations. [PyTorch is well maintained and has been picking up in volume over the years since 2017](https://github.com/pytorch/pytorch/graphs/contributors).
* [Transformers](https://pypi.org/project/transformers/) was started [recently in 2019](https://github.com/huggingface/transformers/graphs/contributors) as a large system of pretrained models for various tasks that can be performed on texts such as classification, information extraction, question answering summarization, translation, text generation and more.

So we change the above to:

```
!pip install deeppavlov==0.14.0 torch==1.7.0 transformers==4.3.0
```

However, we find that this also leads to an error, 

```
"ERROR: Command errored out with exit status 1: python setup.py egg_info Check the logs for full command output."
```
So we instead use:

```
~python -m pip install deeppavlov==0.14.0 torch==1.7.0 transformers==4.3.0
```
Which leads to the same error. Upon closer inspection of the logs we see that we are getting the error after [uvloop](https://pypi.org/project/uvloop/)

So we can try manually installing [uvloop from github source](https://pypi.org/project/uvloop/), however this does not work. From some reading online, it appears that uvloop is dependant upon Cython. Attempting to install Cython leads to a message which says, "Cython already installed."

I suspect part of what we may be dealing with here is that the default CoLab Python is 3.6, whereas a lot of these newer packages may require a newer version of Python, for example 3.9.  This [Stackexchange ](https://stackoverflow.com/questions/60775160/install-python-3-8-kernel-in-google-colaboratory) article goes through how to install and upgrade Python to 3.8 rather than 3.6. We utilized that code, but rather than using 3.8 we instead put Python 3.9.

However, when attempting to leverage a newer Python version and then re-installing uvloop, we still get an error.

Instead of going forward in time, it would be better to go back in time. I believe from a Facebook post that I found that this version of this notebook came out in March, 2020. So, we could roll back our version of uvloop to March 2020 and prior to get it working.  Prior to March 2020, the [most recent version of uvloop was November 5th, 2019](https://pypi.org/project/uvloop/#history) at version 0.14.0. When we attempt to install this on CoLab, it works!

After we attempt the 'latest' module installation above, we see several errors that I might need to deal with later.

```
ERROR: umap-learn 0.5.0 has requirement scikit-learn>=0.22, but you'll have scikit-learn 0.21.2 which is incompatible.
ERROR: tensorflow 2.4.1 has requirement numpy~=1.19.2, but you'll have numpy 1.18.0 which is incompatible.
ERROR: google-colab 1.0.0 has requirement pandas~=1.1.0; python_version >= "3.0", but you'll have pandas 0.25.3 which is incompatible.
ERROR: google-colab 1.0.0 has requirement requests~=2.23.0, but you'll have requests 2.22.0 which is incompatible.
ERROR: fbprophet 0.7.1 has requirement pandas>=1.0.4, but you'll have pandas 0.25.3 which is incompatible.
ERROR: datascience 0.10.6 has requirement folium==0.2.1, but you'll have folium 0.8.3 which is incompatible.
ERROR: albumentations 0.1.12 has requirement imgaug<0.2.7,>=0.2.5, but you'll have imgaug 0.2.9 which is incompatible.
Installing col
```
From this point however, I'm going to move on.

### Debugging Module Imports

The next part of the notebook gives us a few imports, which result in bugs when run.

```
from typing import List, Optional, Collection

import torch
from transformers import BertTokenizer, BertForMaskedLM

from deeppavlov import build_model
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
```
Which leads to no errors at this time.

### Registering the bert_encoder

Using a pre-trained BertTokenizer model, we batch text pairs and return our tensors and attention masks.

```
@register('bert_encoder')
class TransformersBertEncoder(Component):
    def __init__(self, pretrained_model: str = 'bert-base-uncased', **kwargs):
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(pretrained_model)
        
    def __call__(self, texts_batch: List[str], text_pairs_batch: Optional[List[str]] = None):
        if text_pairs_batch is not None:
            data = list(zip(texts_batch, text_pairs_batch))
        else:
            data = texts_batch
        
        res = self.tokenizer.batch_encode_plus(data, pad_to_max_length=True, add_special_tokens=True, return_tensors='pt', return_attention_masks=True)
        return res['input_ids'], res['attention_mask'], res['token_type_ids']
```
### Generate a Follow Up of an Initial Text

```
@register('bert_generator')
class TransformersBertGenerator(Component):
    def __init__(self, pretrained_model: str = 'bert-base-uncased',
                 max_generated_tokens: int = 15,
                 mask_token_id: int = 103, sep_token_id: int = 102, pad_token_id: int = 0, **kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: BertForMaskedLM = BertForMaskedLM.from_pretrained(pretrained_model).to(self.device)
        self.max_generated_tokens = max_generated_tokens
        
        self.mask_tensor = torch.tensor(mask_token_id, device=self.device)
        self.sep_tensor = torch.tensor(sep_token_id, device=self.device)
        self.pad_tensor = torch.tensor(pad_token_id, device=self.device)
        
    @staticmethod
    def _sample(prediction_scores: torch.Tensor):
        # return prediction_scores.argmax(dim=-1)
        probas = torch.nn.functional.softmax(prediction_scores[:, 0], dim=-1)
        return torch.multinomial(probas, num_samples=1)
    
    def __call__(self, input_ids: torch.Tensor, attention_masks: torch.Tensor, token_type_ids: torch.Tensor):
        input_ids = input_ids.to(self.device)
        attention_masks = attention_masks.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        
        batch_size = torch.tensor(len(input_ids), device=self.device)
        with torch.no_grad():
            # indexes of all tokens that will be genertated
            mask_indexes = torch.arange(self.max_generated_tokens, device=self.device).expand([batch_size, -1]) + attention_masks.sum(dim=1).unsqueeze(1) - 1
            
            # expand attention masks and token types matrixes to accomodate for addtitional tokens
            attention_masks = torch.cat([attention_masks, torch.zeros([batch_size, self.max_generated_tokens], device=self.device, dtype=int)], dim=1)
            attention_masks.scatter_(1, mask_indexes+1, 1)
            token_type_ids = torch.cat([token_type_ids, torch.ones([batch_size, self.max_generated_tokens], device=self.device, dtype=int)], dim=1)
            
            # expand token ids matrixes with paddings
            input_ids = torch.cat([input_ids, self.pad_tensor.expand(batch_size, self.max_generated_tokens)], dim=1)
            # insert [MASK] and [SEP] tokens
            input_ids.scatter_(1, mask_indexes, self.mask_tensor)
            input_ids.scatter_(1, attention_masks.sum(dim=1).unsqueeze(1)-1, self.sep_tensor)
            
            # fill in masks one by one
            for i in range(self.max_generated_tokens):
                indexes = mask_indexes[:, i:i+1]
                prediction_scores = self.model.forward(input_ids, attention_masks, token_type_ids)[0]
                mask_predictions = prediction_scores.gather(1, indexes.unsqueeze(-1).expand((-1, -1, prediction_scores.shape[-1])))
                input_ids.scatter_(1, indexes, self._sample(mask_predictions))
        return input_ids.cpu().numpy()
```
### Decoding output ids into tokens for second sentences.

```
@register('bert_decoder')
class TransformersBertDecoder(Component):
    def __init__(self, tokenizer: BertTokenizer, stopwords: Collection[str] = ('.', '?', '!'), **kwargs):
        self.tokenizer = tokenizer
        self.stopwords = set(stopwords)
    
    def __call__(self, ids_batch: List[List[int]]):
        result = []
        
        for tokens_ids in ids_batch:
            all_tokens = iter(self.tokenizer.convert_ids_to_tokens(tokens_ids))
            # skip the first part
            for token in all_tokens:
                if token == '[SEP]':
                    break
            tokens = []
            # take tokens until finding `[SEP]` or one of the stopwords
            for token in all_tokens:
                if token == '[SEP]':
                    break
                tokens.append(token)
                if token in self.stopwords:
                    break
            result.append(' '.join(tokens).replace(' ##', '').replace('##', ''))
            
        return result
```
### Using DeepPavlov to Configure the Pipeline

```
config = {
    'chainer': {
        'in': ['texts', 'suggestions'],
        'pipe': [
            {
                'class_name': 'bert_encoder',
                'id': 'encoder',
                'pretrained_model': '{PRETRAINED_MODEL}',
                'in': ['texts', 'suggestions'],
                'out': ['input_ids', 'attention_masks', 'token_type_ids']
            },
            {
                'class_name': 'bert_generator',
                'pretrained_model': '{PRETRAINED_MODEL}',
                'max_generated_tokens': 10,
                'mask_token_id': '#encoder.tokenizer.mask_token_id',
                'sep_token_id': '#encoder.tokenizer.sep_token_id',
                'pad_token_id': '#encoder.tokenizer.pad_token_id',
                'in': ['input_ids', 'attention_masks', 'token_type_ids'],
                'out': ['output_ids']
            },
            {
                'class_name': 'bert_decoder',
                'tokenizer': '#encoder.tokenizer',
                'stopwords': ['.', '!', '?'],
                'in': ['output_ids'],
                'out': ['result']
            }
        ],
        'out': ['result']
    },
    'metadata': {
        'variables': {
            'PRETRAINED_MODEL': 'bert-base-uncased'
        }
    }
}
```
### Initializing the Model to Test on Inputs

```
dp_model = build_model(config)
```
### Generating Texts

```
texts = [
    'DeepPavlov is an open source conversational AI framework.',
    'The inference can speed up multiple times if you switch from CPU to GPU usage.',
    'It is a period of civil war.'
]
suggestions = [
    'I think that it',
    'No result is an expected behavior and it means',
    'Rebel spaceships, striking from a hidden base, have won their first victory against'
]

results = dp_model(texts, suggestions)

print(*zip(texts, results), sep='\n')
```

When we run the above, we get the following errors:

```
/usr/local/lib/python3.6/dist-packages/transformers/tokenization_utils_base.py:2155: FutureWarning: The `pad_to_max_length` argument is deprecated and will be removed in a future version, use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or use `padding='max_length'` to pad to a max length. In this case, you can give a specific length with `max_length` (e.g. `max_length=45`) or leave max_length to None to pad to the maximal input size of the model (e.g. 512 for Bert).
  FutureWarning,
Keyword arguments {'return_attention_masks': True} not recognized.

---------------------------------------------------------------------------

RuntimeError                              Traceback (most recent call last)

<ipython-input-24-d6be8abbbfdf> in <module>()
     10 ]
     11 
---> 12 results = dp_model(texts, suggestions)
     13 
     14 print(*zip(texts, results), sep='\n')
```
So from here, we take out the parameter, "return_attention_masks=True)" from:

```
res = self.tokenizer.batch_encode_plus(data, pad_to_max_length=True, add_special_tokens=True, return_tensors='pt',return_attention_masks=True)
```
After this we get:

```
RuntimeError: Index tensor must have the same number of dimensions as src tensor
```
referring to:

```
input_ids.scatter_(1, mask_indexes, self.mask_tensor)
```
This appears to be a PyTorch error.

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