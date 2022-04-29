***[CSE 256 SP 2022: assignment 3: Comparing Language Models ]***

The code is built based on the given code:
[running] By running python data.py, you can see the model result. 

[testing] You can test the code or the results of the experience by changing the backoff value and smoothing alpha value in lm.py Trigarm class, __init__ function.

[sampling] You can get sampling sentence with specific perfix by changing the sent in sample_sentence in generator.py

You may need to first run:
 > pip install tabulate

Then should be able to run :
 > python data.py


***[ Files ]***

There are three python files in this folder:

- (lm.py): This file describes the higher level interface for a language model, and contains functions to train, query, and evaluate it. An implementation of a simple back-off based unigram model is also included, that implements all of the functions
of the interface.

- (generator.py): This file contains a simple word and sentence sampler for any language model. Since it supports arbitarily complex language models, it is not very efficient. If this sampler is incredibly slow for your language model, you can consider implementing your own (by caching the conditional probability tables, for example, instead of computing it for every word).

-  (data.py): The primary file to run. This file contains methods to read the appropriate data files from the archive, train and evaluate all the unigram language models (by calling “lm.py”), and generate sample sentences from all the models (by calling  “generator.py”). It also saves the result tables into LaTeX files.



*** [ Acknowledgements ]***
Python files adapted from a similar assignment by Sameer Singh