import json
import logging


class TrainingConfig:
    def __init__(self, n_epoch, learning_rate, model_name, checkpoint_dir, early_stop, training_log, pos_weight, test_only):
        """

        :param n_epoch:
        :param learning_rate:
        :param model_name:
               Base model, no matter pre-train or fine-tune  (e.g. HAN)
        :param checkpoint_dir: directory to save new checkpoint.
                               This is ignored when using traditional ML, because we will not store traditional model
        :param early_stop:
        :param training_log:
        :param source_model_checkpoint_dir: If model can be load from checkpoint_dir, this is ignored.
                                            Otherwise, partially restore model from source model
        :param target_task_model_name: Only used when fine-tune.
                                       When using transfer learning, it should be the transfer model (e.g. FullyConnect)
                                       When using traditional ML, we currently support SVM
        """
        self.n_epoch = n_epoch
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.early_stop = early_stop
        self.training_log = training_log

        if test_only is None:
            self.test_only = False
        else:
            self.test_only = test_only

        self.pos_weight = pos_weight


    @staticmethod
    def load_from_json(json_path):
        with open(json_path) as f:
            d = json.load(f)
            logging.info("TrainingConfig: {}".format(d))

        return TrainingConfig(
            n_epoch=d["n_epoch"],
            learning_rate=d["learning_rate"],
            model_name=d["model_name"],
            checkpoint_dir=d["checkpoint_dir"],
            early_stop=d["early_stop"],
            training_log=d["training_log"],
            pos_weight=d.get("pos_weight"),
            test_only=d.get("test_only")
        )
