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
        self.ngram_counts = [defaultdict(int) for i in xrange(self.n)]
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

    def get_emission(self):
        for word, ne_tag in self.emission_counts:
            counts = self.emission_counts[(word, ne_tag)]
            ne_tag_count = self.ngram_counts[0][ne_tag]
            emission = counts/ne_tag_count
            self.emision_param[(word, ne_tag)] = emission
    
    def get_trigram(self, a, b, c):
        trigram_count = self.ngram_counts[2][(a, b, c)]
        bigram_count = self.ngram_counts[1][(a, b)]
        return trigram_count/bigram_count
        
                
    def viterbi(self, sentence):
        def find_set(idx):
            if idx in range(1, len(sentence)+1):
                return self.all_states
            elif idx == 0 or idx == -1:
                return {'*'}

        def pi_viterbi(k, u, v, sentence):
            prob_list = defaultdict(float)
            if k == 0 and u == '*' and v == '*':
                return (1., '*')
            else:
                for w in find_set(k-2):
                    # params in pi_viterbi
                    prev_prob = pi_viterbi(k-1, w, u, sentence)[0]
                    q_prob = self.get_trigram(w, u, v)
                    e_prob = self.get_emission(sentence[k], v)
                    prob = prev_prob * q_prob * e_prob
                    prob_list[(w, u)] = prob
                max_prob = max(prob.items(), key=lambda x: x[1])
                return max_prob[1], max_prob[0][0]
        
        bp = defaultdict(str)
        k = len(sentence)
        U = ''
        V = ''
        P = 0.

        for idx in range(1, k+1):
            prob_list = defaultdict(float)
            for u in find_set(idx-1):
                for v in find_set(idx):
                    max_prob, w = pi_viterbi(idx, u, v, sentence)
                    prob_list[(idx, u, v)] = max_prob
                    bp[(idx, u, v)] = w
            max_tag = max(prob.items(), key=lambda x: x[1])
            #TODO: might have bugs
            bp[(i, max_tag[0][-2], max_tag[0][-1])] = max_tag[0][-2]

            U = max_tag[0][-2]
            V = max_tag[0][-1]
            P = max_tag[1]
            

        #stores (word:tag) in this whole sentence
        sentence_with_tag = defaultdict(str)

        sentence_with_tag = list()
        backpointer = defaultdict(str)
        tags = defaultdict(str)
        k = len(sentence)
        u_glob = ''
        v_glob = ''
        glob = 0.
        for i in range(1, k+1):
            prob = defaultdict(float)
            for u in find_set(i-1):
                for v in find_set(i):
                    value, w = pi_viterbi(i,u,v,sentence)
                    prob[(i,u,v)] = value
                    backpointer[(i,u,v)] = w
            max_tuple = max(prob.items(), key=lambda x: x[1])
            backpointer[(i, max_tuple[0][1], max_tuple[0][-1])] = max_tuple[0][1] # bp (k,u,v)= tag w
    
            #sentence_with_tag.append(max_tuple[0][-1])
            u_glob = max_tuple[0][-2]
            v_glob = max_tuple[0][-1]
            glob = max_tuple[1]
            #print ('Max',max_tuple)
        tags[k-1] = u_glob
        tags[k] = v_glob
        for i in range((k-2),0,-1):
            tag = backpointer[tuple(((i+2),tags[i+1],tags[i+2]))]
            tags[i]=tag
    
        tag_list=list()
        for i in range(1,len(tags)+1):
            tag_list.append(tags[i])
    
        #tag list as results
        return tag_list
 

def usage():
    print ("""
    python count_freqs.py [input_file] > [output_file]
        Read in a gene tagged training input file and produce counts.
    """)

if __name__ == "__main__":

    if len(sys.argv)!=3: # Expect exactly one argument: the training data file
        usage()
        sys.exit(2)

    try:
        input = open(sys.argv[1],"r")
        dev = open(sys.argv[2],"r")
    except IOError:
        sys.stderr.write("ERROR: Cannot read inputfile %s.\n" % arg)
        sys.exit(1)
    
    # Initialize a trigram counter
    counter = Hmm(3)
    # Collect counts
    counter.train(input)
    # replace infrequent words
    #counter.replace_infrequent()
    # replace improve infrequent words
    counter.improve_baseline()
    # Write the counts
    #counter.write_counts(sys.stdout)
    #counter.dev_output(dev, sys.stdout)
    counter.dev_improve(dev, sys.stdout)
    


    



