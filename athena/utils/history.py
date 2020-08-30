class History:
    def __init__(self, train_losses, train_accs, test_losses, test_accs):
        self.test_losses = test_losses
        self.test_accs = test_accs
        self.train_losses = train_losses
        self.train_accs = train_accs