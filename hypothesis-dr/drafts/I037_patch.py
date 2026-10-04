"""Apply I037: train the whitening bias only during epoch 0. Afterwards the forward uses bias.detach() (a module flag
dynamo guards on), so the compiled backward has no gradient path into the whitening output and skips the input-gradient
of block 1's first conv. Both graph variants are compiled/warmed up in build()."""
p = "submissions/arena/submission.py"
s = open(p).read()
def rep(a, b):
    global s
    assert a in s, a
    s = s.replace(a, b)
rep('''    "whiten_bias_epochs": 3,''', '''    "whiten_bias_epochs": 1,''')
rep('''        self.head = nn.Linear(widths[2], num_classes, bias=False)
        self.scale = scale''', '''        self.head = nn.Linear(widths[2], num_classes, bias=False)
        self.scale = scale
        self.whiten_bias_grad = True''')
rep('''        x = F.silu(self.whiten(x))''', '''        bias = self.whiten.bias if self.whiten_bias_grad else self.whiten.bias.detach()
        x = F.silu(F.conv2d(x, self.whiten.weight, bias))''')
rep('''    for _ in range(3):
        train_step(state, opt, x, y)''', '''    for bias_grad in (True, False):
        model.whiten_bias_grad = bias_grad
        for _ in range(3):
            train_step(state, opt, x, y)
    model.whiten_bias_grad = True''')
rep('''        whiten_on = epoch < hyp["whiten_bias_epochs"]
''', '''        whiten_on = epoch < hyp["whiten_bias_epochs"]
        model.whiten_bias_grad = whiten_on
''')
rep('''    model.eval()
    return model''', '''    model.whiten_bias_grad = True
    model.eval()
    return model''')
open(p, "w").write(s)
