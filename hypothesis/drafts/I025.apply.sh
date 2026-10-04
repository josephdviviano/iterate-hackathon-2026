git show 45b511c:submissions/arena/submission.py > submissions/arena/submission.py
sed -i 's/nn.GELU()/nn.GELU(approximate="tanh")/g' submissions/arena/submission.py
