Assignment 1, Spring 2022 at UCSD

Welcome to Statistical NLP, Spring 2022!

We'll be using Python throughout the course. If you've got a good Python setup already, great! But make sure that it is at least Python version 3.5. If not, the easiest thing to do is to make sure you have at least 3GB free on your computer and then to head over to (https://www.anaconda.com/download/) and install the Python 3 version of Anaconda. It will work on any operating system.

After you have installed conda, close any open terminals you might have. Then open a new terminal.

Then run:
 > python sentiment.py

The feature engineering methods were implemented in read_files function. If you want to change to different preprocessing method, remove the noted code. Basic text preprocess is in preporcess function. Stemming was impelemented in stemming function and lemmatization was implemented in lemmatization, which you can change the tokenizer in TFIDF to the function you need. Regularization parameter C can be changed in classify.py.