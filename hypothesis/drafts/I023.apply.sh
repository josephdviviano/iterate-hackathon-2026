git show ed8dc49:submissions/arena/submission.py > submissions/arena/submission.py
python drafts/I021.apply.py
sed -i 's/"epochs": 9.0,/"epochs": 8.0,/; s/"res_schedule": \[\[0.4, 24\], \[0.7, 28\]\],/"res_schedule": [[0.5, 24]],/' submissions/arena/submission.py
