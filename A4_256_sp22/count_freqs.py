#! /usr/bin/python

import sys
from collections import defaultdict
import math
import copy

"""
Count n-gram frequencies in a data file and write counts to
stdout. 
"""

def simple_conll_corpus_iterator(corpus_file):
    """
    Get an iterator object over the corpus file. The elements of the
    iterator contain (word, ne_tag) tuples. Blank lines, indicating
    sentence boundaries return (None, None).
    """
    l = corpus_file.readline()
    while l:
        line = l.strip()
        if line: # Nonempty line
            # Extract information from line.
            # Each line has the format
            # word pos_tag phrase_tag ne_tag
            fields = line.split(" ")
            ne_tag = fields[-1]
            #phrase_tag = fields[-2] #Unused
            #pos_tag = fields[-3] #Unused
            word = " ".join(fields[:-1])
            yield word, ne_tag
        else: # Empty line
            yield (None, None)                        
        l = corpus_file.readline()

def sentence_iterator(corpus_iterator):
    """
    Return an iterator object that yields one sentence at a time.
    Sentences are represented as lists of (word, ne_tag) tuples.
    """
    current_sentence = [] #Buffer for the current sentence
    for l in corpus_iterator:        
            if l==(None, None):
                if current_sentence:  #Reached the end of a sentence
                    yield current_sentence
                    current_sentence = [] #Reset buffer
                else: # Got empty input stream
                    sys.stderr.write("WARNING: Got empty input file/stream.\n")
                    raise StopIteration
            else:
                current_sentence.append(l) #Add token to the buffer

    if current_sentence: # If the last line was blank, we're done
        yield current_sentence  #Otherwise when there is no more token
                                # in the stream return the last sentence.

def get_ngrams(sent_iterator, n):
    """
    Get a generator that returns n-grams over the entire corpus,
    respecting sentence boundaries and inserting boundary tokens.
    Sent_iterator is a generator object whose elements are lists
    of tokens.
    """
    for sent in sent_iterator:
         #Add boundary symbols to the sentence
         w_boundary = (n-1) * [(None, "*")]
         w_boundary.extend(sent)
         w_boundary.append((None, "STOP"))
         #Then extract n-grams
         ngrams = (tuple(w_boundary[i:i+n]) for i in range(len(w_boundary)-n+1))
         for n_gram in ngrams: #Return one n-gram at a time
            yield n_gram        


class Hmm(object):
    """
    Stores counts for n-grams and emissions. 
    """

    def __init__(self, n=3):
        assert n>=2, "Expecting n>=2."
        self.n = n
        self.emission_counts = defaultdict(int)
        self.ngram_counts = [defaultdict(int) for i in range(self.n)]
        self.all_states = set() 
        self.emision_param = defaultdict(float)
        self.trigram_param = defaultdict(float)

    def train(self, corpus_file):
        """
        Count n-gram frequencies and emission probabilities from a corpus file.
        """
        ngram_iterator = \
            get_ngrams(sentence_iterator(simple_conll_corpus_iterator(corpus_file)), self.n)

        for ngram in ngram_iterator:
            #Sanity check: n-gram we get from the corpus stream needs to have the right length
            assert len(ngram) == self.n, "ngram in stream is %i, expected %i" % (len(ngram, self.n))

            tagsonly = tuple([ne_tag for word, ne_tag in ngram]) #retrieve only the tags            
            for i in range(2, self.n+1): #Count NE-tag 2-grams..n-grams
                self.ngram_counts[i-1][tagsonly[-i:]] += 1
            
            if ngram[-1][0] is not None: # If this is not the last word in a sentence
                self.ngram_counts[0][tagsonly[-1:]] += 1 # count 1-gram
                self.emission_counts[ngram[-1]] += 1 # and emission frequencies

            # Need to count a single n-1-gram of sentence start symbols per sentence
            if ngram[-2][0] is None: # this is the first n-gram in a sentence
                self.ngram_counts[self.n - 2][tuple((self.n - 1) * ["*"])] += 1

    def write_counts(self, output, printngrams=[1,2,3]):
        """
        Writes counts to the output file object.
        Format:

        """
        # First write counts for emissions
        for word, ne_tag in self.emission_counts:
            output.write("%i WORDTAG %s %s\n" % (self.emission_counts[(word, ne_tag)], ne_tag, word))
        # Then write counts for all ngrams
        for n in printngrams:            
            for ngram in self.ngram_counts[n-1]:
                ngramstr = " ".join(ngram)
                output.write("%i %i-GRAM %s\n" %(self.ngram_counts[n-1][ngram], n, ngramstr))

    def replace_infrequent(self):
        emission_copy = copy.deepcopy(self.emission_counts)
        for word, _ in emission_copy:
            gene_count = self.emission_counts.get((word, "I-GENE"), 0)
            o_count = self.emission_counts.get((word, "O"), 0)
            if gene_count + o_count < 5:
                self.emission_counts[("_RARE_", "I-GENE")] = self.emission_counts.get(("_RARE_", "I-GENE"), 0)+gene_count
                self.emission_counts[("_RARE_", "O")] = self.emission_counts.get(("_RARE_", "O"), 0)+o_count
                if (word, "I-GENE") in self.emission_counts:
                    del self.emission_counts[(word, "I-GENE")]
                if (word, "O") in self.emission_counts:
                    del self.emission_counts[(word, "O")]
    
    def improve_baseline(self):
        emission_copy = copy.deepcopy(self.emission_counts)
        for word, _ in emission_copy:
            gene_count = self.emission_counts.get((word, "I-GENE"), 0)
            o_count = self.emission_counts.get((word, "O"), 0)
            if gene_count + o_count < 5:
                if word.isupper():
                    self.emission_counts[("_RARE_Upper", "I-GENE")] = self.emission_counts.get(("_RARE_Upper", "I-GENE"), 0)+gene_count
                    self.emission_counts[("_RARE_Upper", "O")] = self.emission_counts.get(("_RARE_Upper", "O"), 0)+o_count
                elif word.isdigit():
                    self.emission_counts[("_RARE_Number", "I-GENE")] = self.emission_counts.get(("_RARE_Number", "I-GENE"), 0)+gene_count
                    self.emission_counts[("_RARE_Number", "O")] = self.emission_counts.get(("_RARE_Number", "O"), 0)+o_count
                else:
                    self.emission_counts[("_RARE_", "I-GENE")] = self.emission_counts.get(("_RARE_", "I-GENE"), 0)+gene_count
                    self.emission_counts[("_RARE_", "O")] = self.emission_counts.get(("_RARE_", "O"), 0)+o_count
                if (word, "I-GENE") in self.emission_counts:
                    del self.emission_counts[(word, "I-GENE")]
                if (word, "O") in self.emission_counts:
                    del self.emission_counts[(word, "O")]

    def dev_output(self, corpus, output_corpus):
        zero_count = 0
        gene_count = 0
        for word, ne_tag in self.emission_counts:
            if ne_tag == "I-GENE":
                gene_count += self.emission_counts[(word, ne_tag)]
            else:
                zero_count += self.emission_counts[(word, ne_tag)]
        rare_gene = self.emission_counts[("_RARE_", "I-GENE")]/gene_count
        rare_zero = self.emission_counts[("_RARE_","O")]/zero_count
        rare_type = ""
        if rare_gene >= rare_zero:
            rare_type = "I-GENE"
        else:
            rare_type = "O"
        line = corpus.readline()
        while line:
            if line:
                fields = line.split(" ")
                word = fields[0].strip('\n')
                #print(word)
                if not len(word):
                    output_corpus.write("\n")
                    line = corpus.readline()
                    continue
                word_gene = 0
                word_zero = 0
                if (word, "I-GENE") in self.emission_counts and (word,'O') in self.emission_counts:
                    word_gene = self.emission_counts[(word, "I-GENE")]/gene_count
                    word_zero = self.emission_counts[(word,"O")]/zero_count
                    word_type = ""
                    if word_gene >= word_zero:
                        word_type = "I-GENE"
                    else:
                        word_type = "O"
                    output_corpus.write(word + " " + word_type +"\n")
                elif (word, "I-GENE") in self.emission_counts:
                    word_type = "I-GENE"
                    output_corpus.write(word + " " + word_type +"\n")
                elif (word,'O') in self.emission_counts:
                    word_type = "O"
                    output_corpus.write(word + " " + word_type +"\n")
                else:
                    output_corpus.write(word + " " + rare_type +"\n")
            line = corpus.readline()


    def dev_improve(self, corpus, output_corpus):
        zero_count = 0
        gene_count = 0
        for word, ne_tag in self.emission_counts:
            if ne_tag == "I-GENE":
                gene_count += self.emission_counts[(word, ne_tag)]
            else:
                zero_count += self.emission_counts[(word, ne_tag)]
        rare_gene = self.emission_counts[("_RARE_", "I-GENE")]/gene_count
        rare_zero = self.emission_counts[("_RARE_","O")]/zero_count
        rare_type = ""
        if rare_gene >= rare_zero:
            rare_type = "I-GENE"
        else:
            rare_type = "O"

        upper_gene = self.emission_counts[("_RARE_Upper", "I-GENE")]/gene_count
        upper_zero = self.emission_counts[("_RARE_Upper", "O")]/zero_count
        upper_type = ""
        if upper_gene >= upper_zero:
            upper_type = "I-GENE"
        else:
            upper_type = "O"

        number_gene = self.emission_counts[("_RARE_Number", "I-GENE")]/gene_count
        number_zero = self.emission_counts[("_RARE_Number", "O")]/zero_count
        number_type = ""
        if number_gene >= number_zero:
            number_type = "I-GENE"
        else:
            number_type = "O"

        line = corpus.readline()
        while line:
            if line:
                fields = line.split(" ")
                word = fields[0].strip('\n')
                #print(word)
                if not len(word):
                    output_corpus.write("\n")
                    line = corpus.readline()
                    continue
                word_gene = 0
                word_zero = 0
                if (word, "I-GENE") in self.emission_counts and (word,'O') in self.emission_counts:
                    word_gene = self.emission_counts[(word, "I-GENE")]/gene_count
                    word_zero = self.emission_counts[(word,"O")]/zero_count
                    word_type = ""
                    if word_gene >= word_zero:
                        word_type = "I-GENE"
                    else:
                        word_type = "O"
                    output_corpus.write(word + " " + word_type +"\n")
                elif (word, "I-GENE") in self.emission_counts:
                    word_type = "I-GENE"
                    output_corpus.write(word + " " + word_type +"\n")
                elif (word,'O') in self.emission_counts:
                    word_type = "O"
                    output_corpus.write(word + " " + word_type +"\n")
                else:
                    if word.isupper():
                        output_corpus.write(word + " " + upper_type +"\n")
                    elif word.isdigit():
                        output_corpus.write(word + " " + number_type +"\n")
                    else:
                        output_corpus.write(word + " " + rare_type +"\n")
            line = corpus.readline()

    def read_counts(self, corpusfile):
        self.n = 3
        self.emission_counts = defaultdict(int)
        self.ngram_counts = [defaultdict(int) for i in range(self.n)]
        self.all_states = set()

        for line in corpusfile:
            parts = line.strip().split(" ")
            count = float(parts[0])
            if parts[1] == "WORDTAG":
                ne_tag = parts[2]
                word = parts[3]
                self.emission_counts[(word, ne_tag)] = count
                self.all_states.add(ne_tag)
            elif parts[1].endswith("GRAM"):
                n = int(parts[1].replace("-GRAM",""))
                ngram = tuple(parts[2:])
                self.ngram_counts[n-1][ngram] = count

    def get_emission(self, word, ne_tag):
        res = 0
        if (word, ne_tag) in self.emission_counts:
            #print(self.emission_counts[(word, ne_tag)])
            counts = self.emission_counts[(word, ne_tag)]
            res = counts/self.ngram_counts[0][ne_tag,]
        return res

    
    def get_trigram(self, a, b, c):
        trigram_count = self.ngram_counts[2][(a, b, c)]
        bigram_count = self.ngram_counts[1][(a, b)]
        return trigram_count/bigram_count
        
                
    def viterbi(self, sentence):
        pi_viterbi = defaultdict(dict)
        backpointer = defaultdict(dict)

        def find_set(idx):
            if idx in range(1, len(sentence)+1):
                return self.all_states
            elif idx == 0 or idx == -1:
                return {'*'}

        pi_viterbi[0]['*', '*'] = 1.
        #backpointer[0]['*', '*'] = ''
        for k in range(1, len(sentence)+1):
            for u in find_set(k-1):
                for v in find_set(k):
                    prob_list = defaultdict(float)
                    for w in find_set(k - 2):
                        prev_p = pi_viterbi[k - 1][w, u]
                        q_p = self.get_trigram(w, u, v)
                        if max([self.get_emission(sentence[k-1], y) for y in find_set(k)]) == 0:
                            e_p = self.emission_counts[("_RARE_", v)]/self.ngram_counts[0][v,]
                        else:
                            e_p = self.get_emission(sentence[k-1], v)
                        prob = prev_p * q_p * e_p
                        prob_list[(w, u)] = prob
                    max_p = max(prob_list.items(), key=lambda x: x[1])
                    pi_viterbi[k][u, v] = max_p[1]
                    backpointer[k][u, v] = max_p[0][0]
        n = len(sentence)
        #print(n)
        final_probs = defaultdict(float)
        #print(len(find_set(1)))
        for u in find_set(n-1):
            for v in find_set(n):
                #print(n, u, v)
                q_p = self.get_trigram(u, v, 'STOP')
                #print(pi_viterbi[n][u, v] * q_p)
                final_probs[(u, v)] = pi_viterbi[n][u, v] * q_p

        max_final = max(final_probs.items(), key=lambda x: x[1])
        u = max_final[0][0]
        v = max_final[0][1]
        tag_predict = defaultdict(str)
        tag_predict[n-1] = u
        tag_predict[n] = v

        for k in range(n-2, 0, -1):
            tmp = backpointer[k+2][tag_predict[k+1], tag_predict[k+2]]
            tag_predict[k] = tmp
        return tag_predict
    
    def call_viterbi(self, corpus, output_corpus):
        sentence = list()
        line = corpus.readline()
        while line:
            fields = line.split(" ")
            word = fields[0].strip('\n')
            if not len(word):
                predict = self.viterbi(sentence)
                #print(len(predict))
                #print(len(sentence))
                for i in range(len(sentence)):
                    #print(sentence[i] + " " + predict[i])
                    output_corpus.write(sentence[i] + " " + predict[i+1] +"\n")
                output_corpus.write("\n")
                sentence = list()
                line = corpus.readline()
                continue
            sentence.append(word)
            line = corpus.readline()

def usage():
    print ("""
    python count_freqs.py [input_file] > [output_file]
        Read in a gene tagged training input file and produce counts.
    """)

if __name__ == "__main__":

    # if len(sys.argv)!=3: # Expect exactly one argument: the training data file
    #     usage()
    #     sys.exit(2)
    #
    # try:
    #     input = open(sys.argv[1],"r")
    #     dev = open(sys.argv[2],"r")
    # except IOError:
    #     sys.stderr.write("ERROR: Cannot read inputfile %s.\n" % arg)
    #     sys.exit(1)
    input = open("gene.counts10", "r")
    dev = open("gene.dev", "r")
    # Initialize a trigram counter
    counter = Hmm(3)
    counter.read_counts(input)
    # Collect counts
    #counter.train(input)
    # replace infrequent words
    #counter.replace_infrequent()
    # replace improve infrequent words
    #counter.improve_baseline()
    counter.call_viterbi(dev, sys.stdout)
    # Write the counts
    #counter.write_counts(sys.stdout)
    #counter.dev_output(dev, sys.stdout)
    #counter.dev_improve(dev, sys.stdout)

    


    



