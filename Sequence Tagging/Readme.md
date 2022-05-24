# CSE 256 A4
this file run on python3
### How to run
basic run, since my function has to have two input
```
python count_freqs.py gene.train gene.dev > gene.counts
```
get rare symobl\
uncomment line 420 in count_freqs.py and run
```
python count_freqs.py gene.train gene.dev > gene.counts
```
get more informative symbol\
uncomment line 421 in count_freqs.py and run
```
python count_freqs.py gene.train gene.dev > gene.counts_improv
```
### Already generated counts
gene.counts
gene.counts_improve

### Generate dev_ouput
basic one\
uncomment line 425 in count_freqs.py and run
```
python count_freqs.py gene.counts gene.dev > gene_dev.p1.out1
```
improved baseline model\
uncomment line 426 in count_freqs.py and run
```
python count_freqs.py gene.counts_improv gene.dev > gene_dev.p1.out1.improve
```
Trigram HMM\
uncomment line 415 and 421 in count_freqs.py and run
```
python count_freqs.py gene.counts gene.dev > gene_dev.output.hmm
```
improved Trigram HMM\
comment the 333 line and 334 line and uncomment the improvement part from line 336 to 342, uncomment line 415 and 421 in count_freqs.py and run
```
python count_freqs.py gene.counts_improv gene.dev > gene_dev.output.hmm.improve
```
### Evaluate the result
evaluate the baseline model 
```
python eval_gene_tagger.py gene.key gene_dev.p1.out1
```
evaluate the improved baseline model
```
python eval_gene_tagger.py gene.key gene_dev.p1.out1.improve
```
evaluate the Trigram HMM model
```
python eval_gene_tagger.py gene.key gene_dev.output.hmm
```
evaluate the improved Trigram HMM model
```
python eval_gene_tagger.py gene.key gene_dev.output.hmm.improve
```

