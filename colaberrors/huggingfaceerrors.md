# HuggingFace Errors on CoLab

### Installing Transformers and Tensorflow

```
!pip install -q git+https://github.com/huggingface/transformers.git
!pip install -q tensorflow==2.1
```
When we run this, we get the error:

"ERROR: tensorflow-probability 0.12.1 has requirement gast>=0.3.2, but you'll have gast 0.2.2 which is incompatible."

This basically means we need to install a dependency.

```
pip install --upgrade pip
pip install gast==0.3.2
```
Note that when running this, we got a message from CoLab noting that we need to restart the runtime.

Even after restarting the runtime and attempting the above installs, we get the same error.

### Install GPT2 Tokenizer

```
import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# add the EOS token as PAD token to avoid warnings
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
```

We get an error:

ImportError: cannot import name 'TFGPT2LMHeadModel' from 'transformers' (unknown location)
