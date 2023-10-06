# Master Computer Vision, Module C1

This repository contains code for the MCV C1 Project. The code can be used to evaluate your results and to
check that the weekly submissions have the proper format.

You can use the evaluation functions in the ```evaluation/``` folder to compute precision, recall, F1, ...,
both at the pixel level and at the window level. Check the ```score_painting_retrieval.py``` script for
an example on how to use the scoring functions.


## Description of the scripts:

### test_submission.py: 
- This script allows the students to test their submissions to minimize the probability of errors.
  It uses a fake GT that is updated every week because the the size of the images has to agree with the ones
  in the week's test set.

  Example of usage of the submission script:

  Assume that 'fake' is the folder with fake ground-truth and 'submission_folder' is the folder where the
  students put the results in Google Drive. 'submission_folder' must contain the folder structure "week?/QST{1,2}/method{1,2}/", similar to the google drive folder where you submit the results. To check the submission for week3, Team 1), the students have to execute:

```
python test_submission.py 3 1 submission_folder fake # Test submission for both QST1 and QST2
```

  another example, for week 5, Team 1:

```
python test_submission_new.py 5 1 submission_folder fake 
```

### score_painting_retrieval.py
- This script is used by the lecturers to perform the scoring. The usage is:

```
# GT files assumed to be on folder W5/qst1_w5
python score_painting_retrieval.py /home/dlcv01/c1-results/week5/QST1 W5/qst1_w5/gt_corresps.pkl --kVal='1,5' --augList=W5/qst1_w5/augmentations.pkl --imaDir=W5/qst1_w5 --gtPicFrames=2020/W5/qst1_w5/frames.pkl
```

### virtualenvs.txt
- Info to create a virtualenv to run this code.


### utils/plot_results.py
- Plot painting retrieval results tables

### utils/print_dict.py
- Print the contents of a .pkl file
