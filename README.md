# word2vec-with-pmi
My own word2vec implementation enriched with point-wise mutual information statistic as a basis of skip-gram co-occurence probability. It uses skipgram and negative sampling architecture along with a feed-forward neural network written in NumPy.

Repository consists of three scripts and a zipped text file with a dump of 10^8 characters of English wikipedia from 2007 as to be used as a test corpus.

## Usage

To create vectors with the script, use:
```bash
python word2vec.py text 100 wiki
```
This will create a model with 100-dimensional word vectors under the model name 'wiki' using 'text' as an input textfile to base on. It takes around 90 minutes to create vectors on the enclosed wiki corpus on a machine with 24gb of RAM, so beware.

After training the model, use:
```bash
python cluster.py wiki
```
This will clusterize the vectors so that finding similar words in a next step can be sped up significantly.

To find similar words with a trained model, use:
```bash
python find_nn.py wiki your_word
```
This will print a list of the most similar words from the corpus along with the cosine similarity measure.

## Author
Scripts written and maintained by Michal Pikusa (pikusa.michal@gmail.com)
