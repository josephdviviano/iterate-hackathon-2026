git show 9efb085:submissions/arena/submission.py > submissions/arena/submission.py
sed -i 's/        update = torch.compile(muon_update, mode="max-autotune-no-cudagraphs")/        update = torch.compile(muon_update)/' submissions/arena/submission.py
