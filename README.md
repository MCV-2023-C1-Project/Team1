# Team1

## Setup Environment

to create the new environment run:

``bash
bash env.sh 
``

in case the environment hasn't been activated rewrite
``
source myenv/bin/activate
``

see that you can customize the path and the name of the environment changing the variables inside the script (Be consistent with that)

## Structure
````
results/
    {method}+{similarity}.pkl
    ...

tools/
    Some foreign tools given by @rmorros

data/
    BBDD/
        bbdd_00000.jpg
        bbdd_00000.png
        bbdd_00000.txt
        ...
    qsd1_w1/
        00000.jpg
        ...
    qsd2_w1/
        00000.jpg
        00000.png
        ...
src/
    common/
        ...
    methods/
        ...
    metrics/
        ...
    preprocessing/
        ...
    utils/
        ...
    ...


main.py
main_test.py
````

## Command Line Options and Usage
This script provides a command-line interface to perform museum painting retrieval based on various methods for computing descriptors and similarity measures. The available options and their combinations are as follows:

Query File Option:

* -qf, --queryfile <file_path>: Specify a file containing ground truth queries for the retrieval process. Default is False.

* -q, --queries <folder_path>: Specify the folder containing the queries for the retrieval process. This option is required.

* --update: Check the descriptors database and update it if there are new images to compute their descriptors.

* --overwrite: Compute and overwrite all descriptors in the database.

* -m, --method <method>: Choose the method to compute descriptors. Available choices are: gray_hist, norm-rg, cummulative, multitile.
Similarity Measure Option:

* -s, --similarity <similarity>: Choose the method to compute similarity. Available choices are: cosine, l1, euc, chi, hellkdis, jensen, histint.
Results Storage Option:


* -k, --k <value>: Specify the value of K to compute the retrieval. Default is 1.

* -nt, --tiles <value>: Specify the number of tiles for local slicing during descriptor computation. Default is 6.


### Run example
``python
python src/main.py -q data/qsd1_w1  -m multitile -qf data/qsd1_w1/gt_corresps.pkl -k 10 --similarity histint --tiles 6 --overwrite
``

