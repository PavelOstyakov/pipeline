class TrainerBase:
    def __init__(self, model, train_data_loader, val_data_loader, epoch_count, optimizer,
                 scheduler, loss_calculator, metric_calculator, print_frequency):
        self.model = model
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.epoch_count = epoch_count
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_calculator = loss_calculator
        self.metric_calculator = metric_calculator
        self.print_frequency = print_frequency

    def train_step(self, input_data, target):
        raise NotImplementedError

    def predict_step(self, input_data):
        pass

    def run_train_epoch(self):
        pass

    def run_val_epoch(self):
        pass

    def run(self):
        pass
