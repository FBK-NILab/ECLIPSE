import os
import time
import logging

formatter = logging.Formatter("{asctime} - {levelname} - {message}", style="{")

def setup_logger(name, log_file, level=logging.DEBUG):

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(logging.DEBUG)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    return logger


class Logger:

    def __init__(self, name: str, logger_params: dict, mode: str = 'train'):
        if mode == "train":
            self.tf_log = logger_params["tf_log"]
            self.name = name
            self.base_folder = None
            if self.tf_log:
                from torch.utils.tensorboard import SummaryWriter
                self.log_dir = os.path.join(self.name, 'logs')
                self.writer = SummaryWriter(self.log_dir)

            self.train_log_file = os.path.join(self.name, 'loss_log.txt')
            self.eval_log_file = os.path.join(self.name, 'eval_log.txt')
            self.test_log_file = os.path.join(self.name, 'test_log.txt')
            self.train_logger = setup_logger(name='train_logger', log_file=self.train_log_file)
            self.val_logger = setup_logger(name='val_logger', log_file=self.eval_log_file)
            self.test_logger = setup_logger(name='test_logger', log_file=self.test_log_file)
            with open(self.train_log_file, "a") as log_file:
                now = time.strftime("%c")
                log_file.write(f'================ Training Log ({now}) ================\n')

            with open(self.eval_log_file, "a") as eval_log_file:
                now = time.strftime("%c")
                eval_log_file.write(f'================ Eval Log ({now}) ================\n')

            with open(self.test_log_file, "a") as test_log_file:
                now = time.strftime("%c")
                test_log_file.write(f'================ Test Log ({now}) ================\n')
        else:
            self.name = name
            self.test_log_file = os.path.join(self.name, 'test_log.txt')
            self.test_logger = setup_logger(name='test_logger', log_file=self.test_log_file)
            with open(self.test_log_file, "a") as test_log_file:
                now = time.strftime("%c")
                test_log_file.write(f'================ Test Log ({now}) ================\n')


    def plot_train(self, errors, step):
        """
        # errors: dictionary of error labels and values
        Parameters
        ----------
        errors
        step

        Returns
        -------

        """
        if self.tf_log:
            for tag, value in errors.items():
                value = float(value)
                self.writer.add_scalar(f"Train/{tag}", value, step)


    def plot_lr(self, lr, step):
        """
        # lr: current learning rate
        Parameters
        ----------
        lr
        step

        Returns
        -------

        """
        if self.tf_log:
            self.writer.add_scalar(f"Train/lr", lr, step)


    def plot_eval(self, eval_stats, step):
        """
        # errors: dictionary of error labels and values
        Parameters
        ----------
        eval_stats
        step

        Returns
        -------

        """
        if self.tf_log:
            for tag, value in eval_stats.items():
                value = float(value)
                self.writer.add_scalar(f"Eval/{tag}", value, step)


    def log_train_stats(self, epoch: int, i: int = None, errors: dict = None):
        """
        errors: same format as |errors| of plotCurrentErrors
        Parameters
        ----------
        epoch
        i
        errors

        Returns
        -------

        """
        message = f'(epoch: {epoch}, iters: {i}) ' if i else f'(epoch: {epoch}) '
        for k, v in errors.items():
            if v != 0:
                message += f'{k}: {v:.7f} '
        self.train_logger.info(message)


    def log_eval_stats(self, epoch, eval_stats):
        """
        log eval stats to stdout and file
        Parameters
        ----------
        epoch
        eval_stats

        Returns
        -------

        """
        message = f'(Eval epoch: {epoch}) '
        for k, v in eval_stats.items():
            if v != 0:
                message += f'{k}: {v:.3f} '
        self.val_logger.info(message)


    def log_test_stats(self, header, test_stats):
        """
        log eval stats to stdout and file
        Parameters
        ----------
        header
        test_stats

        Returns
        -------

        """
        message = f"{header}\n"
        for k, v in test_stats.items():
            if v != 0:
                message += f'{k}: {v:.3f} '
        self.test_logger.info(message)
