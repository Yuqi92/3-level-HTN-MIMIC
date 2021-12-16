from data_loader import DataLoaderConfig, DataLoader
from predict_result_writer import PredictResultWriter
import logging
import tensorflow as tf
from training_config import TrainingConfig
import sys
import os
from models import *
import datetime
import time


def main(argv):

    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG)
    start_time = time.time()

    if len(argv) != 3:
        logging.warning("Usage: python train.py <training_config_file> <data_config_file> <model_config_file>")
        return

    training_config_file = argv[0]
    data_config_file = argv[1]
    model_config_file = argv[2]


    # load training config
    training_config = TrainingConfig.load_from_json(training_config_file)

    # load data config from json configuration file
    data_config = DataLoaderConfig.load_from_json(data_config_file)
    # load data from module DataLoader with config file
    data_loader = DataLoader(data_config)

    # get the train and dev data set
    training_set = data_loader.get_dataset("training")
    dev_set = data_loader.get_dataset("dev")
    test_set = data_loader.get_dataset("test")

    n_label = data_config.n_label
    label_list = data_config.label_list


    strategy = tf.distribute.MirroredStrategy()

    training_set = strategy.experimental_distribute_dataset(training_set)
    dev_set = strategy.experimental_distribute_dataset(dev_set)
    test_set = strategy.experimental_distribute_dataset(test_set)

    # Start to build model
    # get the specific model class. for example, "HAN + Model" refers to the HANModel class,
    # also the same with config class
    model_class = eval(training_config.model_name + "Model")
    model_config_class = eval(training_config.model_name + "ModelConfig")

    with strategy.scope():
        model = model_class(
            model_config_class.load_from_json(model_config_file),
            data_config
        )
        # lr_schedule = tf.keras.experimental.CosineDecay(training_config.learning_rate,
        #                                                         decay_steps=1,
        #                                                         alpha=0.0,
        #                                                         name=None)

        # optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.9, beta_2=0.999,
        #                              epsilon=1e-6)
        optimizer = tf.keras.optimizers.Adam(learning_rate=training_config.learning_rate)

        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint,
            training_config.checkpoint_dir,
            max_to_keep=training_config.early_stop + 5  # save the latest eight model ckpts
        )

        checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        logging.info("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        logging.info("Initializing from scratch.")

    # write the probablity and label file for development and test set
    logging.info("Initialize predict result writer")
    validation_path = os.path.join(training_config.checkpoint_dir, "dev_prediction_result.txt")
    test_path = os.path.join(training_config.checkpoint_dir, "test_prediction_result.txt")
    validation_result_writer = PredictResultWriter(validation_path, label_list)
    test_result_writer = PredictResultWriter(test_path, label_list)


    # training loss and evaluation metrics
    # with mirrored_strategy.scope():
    epoch_loss_metrics = tf.keras.metrics.Mean()
    epoch_auc_metrics = [tf.keras.metrics.AUC(curve='ROC') for _ in range(n_label)]

    # dev loss and evaluation metrics
    dev_loss_metrics = tf.keras.metrics.Mean()
    dev_auc_metrics = [tf.keras.metrics.AUC(curve='ROC') for _ in range(n_label)]

    # test loss and evaluation metrics
    test_loss_metrics = tf.keras.metrics.Mean()
    test_auc_metrics = [tf.keras.metrics.AUC(curve='ROC') for _ in range(n_label)]

    # tensorboard for checking training status

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = training_config.checkpoint_dir + '/' + current_time + '/train'
    dev_log_dir = training_config.checkpoint_dir + '/' + current_time + '/dev'
    test_log_dir = training_config.checkpoint_dir + '/' + current_time + '/test'

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    dev_summary_writer = tf.summary.create_file_writer(dev_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    def get_loss(x, y):
        logits = model(x, training=True)
        if training_config.pos_weight is None:
            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.cast(y, tf.float32),
                    logits=logits
                ),
                axis=1
            )
        else: # imbalanced class: assign positive weights to the positive class loss
            loss = tf.reduce_mean(
                tf.nn.weighted_cross_entropy_with_logits(
                    labels=tf.cast(y, tf.float32),
                    logits=logits,
                    pos_weight=training_config.pos_weight
                ),
                axis=1
            )
        return loss, logits

    # define a train step with gradient method, the goal is to get the train loss/metrics and optimize
    # 1. get the predicted logit
    # 2. calculate the loss between gold label
    # 3. optimize the loss
    # 4. append the current loss to the total loss
    # 5. calculate the metrics for each label

    def train_step(x, y):
        with tf.GradientTape() as tape:
            loss, logits = get_loss(x, y)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # metrics
        epoch_loss_metrics(loss)
        sigmoid_logits = tf.sigmoid(logits)
        for i in range(n_label):
            epoch_auc_metrics[i].update_state(
                y_pred=sigmoid_logits[:, i],
                y_true=y[:, i]
            )

    # distributed training
    @tf.function
    def distributed_train_step(x, y):
        strategy.run(train_step, args=(x,y))

    # similarly, define the validation/test step and the goal is to get the val/test loss/metrics
    def validation_test_step(x, y, pid, loss_metrics, auc_metrics):
        loss, logits = get_loss(x, y)
        # metrics
        loss_metrics(loss)
        sigmoid_logits = tf.sigmoid(logits)
        for i in range(n_label):
            auc_metrics[i].update_state(
                y_pred=sigmoid_logits[:, i],
                y_true=y[:, i]
            )
        return pid, sigmoid_logits, y


    def distributed_validation_test_step(x, y, pid, loss_metrics, auc_metrics, predict_result_writer):
        pr_pid, pr_sigmoid_logits, pr_y = strategy.run(
            validation_test_step, args=(x, y, pid, loss_metrics, auc_metrics))

        #  for multiple single gpu option
        if strategy.num_replicas_in_sync > 1:
            pr_pid = pr_pid.values
            pr_sigmoid_logits = pr_sigmoid_logits.values
            pr_y = pr_y.values
            for i in range(len(pr_pid)):
                predict_result_writer.write(pr_pid[i], pr_sigmoid_logits[i], pr_y[i])
        else:
            predict_result_writer.write(pr_pid, pr_sigmoid_logits, pr_y)


    # function to log the training status at defined epoch, metrics using here: AUC
    def log_status(log_type, batch, epoch, auc_metrics, loss_metrics, reset_metrics=False):
        loss_result = loss_metrics.result()
        auc_value_list = []
        for (M, i) in enumerate(auc_metrics):
            auc_value_list.append("{}:{}".format(label_list[M], i.result().numpy()))
        auc_value = ",".join(auc_value_list)
        s = "{} -- Epoch: {}, Batch: {}, Loss: {}, AUC: [{}]".format(
            log_type, epoch, batch, loss_result, auc_value
        )
        logging.info(s)


        if training_config.training_log:
            with open(training_config.training_log, "a") as f:
                f.write(s + "\n")
        if reset_metrics:
            for i in auc_metrics:
                i.reset_states()
            loss_metrics.reset_states()
        return loss_result

    # tensorboard summary writing
    def tensorboard_log(summary_writer, loss_metrics, auc_metrics, batch, epoch, train=False, dev=False, test=False):
        loss_result = loss_metrics.result()
        with summary_writer.as_default():
            if train:
                tf.summary.scalar('train_loss', loss_result, step=batch)
            if dev:
                tf.summary.scalar('dev_loss', loss_result, step=epoch)
                for (M, i) in enumerate(auc_metrics):
                    scalar_name = "dev_auc_" + label_list[M]
                    scalar_data = i.result()
                    tf.summary.scalar(scalar_name, scalar_data, step=epoch)
            if test:
                tf.summary.scalar('test_loss', loss_result, step=0)

            summary_writer.flush()

    best_loss = None
    n_decrease_dev = 0
    if not training_config.test_only:
        for epoch in range(training_config.n_epoch):
            batch = 0
            for x, y, _ in training_set:
                distributed_train_step(x, y)
                if (epoch == 0)  and (batch == 0):
                    # logging model architecture at the beginning
                    model.summary(print_fn=logging.info)

                # logging every 100 batches/samples
                if batch % 100 == 0:
                    log_status("In epoch", batch, epoch, epoch_auc_metrics, epoch_loss_metrics)
                    tensorboard_log(train_summary_writer, epoch_loss_metrics, None, batch, None, train=True, dev=False, test=False)
                batch += 1

            # For every epoch, reset the metrics/loss and log the metrics and loss
            log_status("End of epoch", batch, epoch, epoch_auc_metrics, epoch_loss_metrics,  reset_metrics=True)

            # save the checkpoint
            save_path = checkpoint_manager.save()
            logging.info("Saved checkpoint for epoch {}: {}".format(epoch, save_path))

            # For each epoch do validation
            logging.info("Start validation")
            validation_result_writer.writer_sep()
            for x, y, subject_id in dev_set:
                distributed_validation_test_step(x, y, subject_id, dev_loss_metrics, dev_auc_metrics, validation_result_writer)

            tensorboard_log(dev_summary_writer, dev_loss_metrics, dev_auc_metrics, None, epoch, train=False, dev=True, test=False)
            new_dev_loss = log_status("Validation", None, epoch, dev_auc_metrics, dev_loss_metrics, reset_metrics=True)

            # Early stopping using dev loss
            if (best_loss is None) or (new_dev_loss < best_loss):
                logging.info("Get better dev loss. Previous best loss: {}. New loss: {}".format(
                    best_loss, new_dev_loss
                ))
                best_loss = new_dev_loss
                n_decrease_dev = 0
            else:
                n_decrease_dev += 1
                logging.info("Get worse dev loss. Previous best loss: {}. New loss: {}. N bad loss: {}".format(
                    best_loss, new_dev_loss, n_decrease_dev
                ))
                if n_decrease_dev >= training_config.early_stop:
                    logging.info("Early stop is set to {}. STOP!!!".format(training_config.early_stop))
                    break

    # finish training, start test
    logging.info("Start test")

    test_result_writer.writer_sep()
    for x, y, subject_id in test_set:
        distributed_validation_test_step(x, y, subject_id, test_loss_metrics, test_auc_metrics, test_result_writer)

    tensorboard_log(test_summary_writer, test_loss_metrics, None, None, None, train=False, dev=False, test=True)
    log_status("Test", None, None, test_auc_metrics, test_loss_metrics, reset_metrics=True)

    validation_result_writer.destroy()
    test_result_writer.destroy()

    # calculate the total executation time
    end_time = time.time()
    exe_time = (end_time - start_time) / 3600
    logging.info("Totally spend {} hours".format(exe_time))



if __name__ == '__main__':
    main(sys.argv[1:])


