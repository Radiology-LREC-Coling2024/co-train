import os
import pandas as pd
import numpy as np
import transformers
import torch
import torch.nn as nn
#from torch.optim import AdamW
from transformers import AdamW
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


# from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import time
import datetime
import random
from sklearn.metrics import (
    confusion_matrix,
    auc,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
)
import math
import sys
from dateutil import tz

from data_processing import organize_labeled_data, organize_unlabeled_data
from dataset import RadiologyLabeledDataset, RadiologyUnlabeledDataset

from torch.utils.tensorboard import SummaryWriter

# add all concluding results to a big excel spread sheet so that I don't have to go through 25 experiments individually 
# Split data into 5 splits and run the code - DONE
# Use validation data to perform early stopping - DONE 

# def split_folds(nfolds, df):
    
#     kf = KFold(n_splits=nfolds, random_state=0, shuffle=True)
    
#     for train_index, test_index in kf.split(df):
#         df_tr_va, df_test = df.iloc[train_index], df[test_index]
#         df_train, df_val = train_test_split(df_tr_va, test_size=0.25, random_state=0)
#         print("TRAIN:", list(df_train.index), "VALIDATION:", list(df_val.index), "TEST:", test_index)
        

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
        self.encoding = "UTF-8"

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush


# Algorithm setup
# 1. Train each model with the different parts of the data
# 2. Predict confident labels for each
# 3. Pass the confident labels to each model
# 4. Calculate test accuracy after each model ... I can implement AUROC and AUPRC here


class SelfTrain:
    def __init__(
        self,
        model1: str,
        model2: str,
        logdir: str,
        train_df,
        test_df,
        val_df,
        unlabeled_df,
        max_length: int,
        batch_size: int,
        num_classes: int,
        learning_rate: float,
        num_epochs: int,
        target: str,
        selftrain_steps: int,
        init_coverage: float,
        all_results_path:str,
    ):
        tzone = tz.gettz("America/Edmonton")
        self.timestamp = (
            datetime.datetime.now().astimezone(tzone).strftime("%Y-%m-%d_%H:%M:%S")
        )

        self.model1 = model1
        self.model2 = model2

        # writer to log information:
        self.logdir = logdir
        self.writer = SummaryWriter(self.logdir)
        self.logger = Logger(os.path.join(self.logdir, self.timestamp + ".log"))
        sys.stdout = self.logger
        sys.stderr = self.logger
        self.results = None

        # assign all of the various data sets that we need
        self.train_df = train_df
        self.test_df = test_df
        self.val_df = val_df
        self.unlabeled_df = unlabeled_df

        # Set training parameters of each separate model:
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.target = target

        # Set co-training parameters:
        self.selftrain_steps = selftrain_steps
        self.init_coverage = init_coverage

        # Track global step counts for metrics
        self.findings_global_step = 0
        self.impressions_global_step = 0
        
        self.all_results_path=all_results_path

    def load_dataset(self, section, df, labeled=True, other_section="", shuffle=True):
        # load data into dataloader + tokenize
        # To-Do: Using self.model1 essentially assumes that the two models used are the same...
        tokenizer = transformers.DistilBertTokenizer.from_pretrained(self.model1)
        if labeled:
            dataset = RadiologyLabeledDataset(
                tokenizer,
                max_length=self.max_length,
                df=df,
                target=self.target,
                text_col=section,
            )
            # if they are labeled, then we want to shuffle the data
            dataloader = DataLoader(
                dataset=dataset, batch_size=self.batch_size, shuffle=shuffle,
            )
        else:
            dataset = RadiologyUnlabeledDataset(
                tokenizer,
                max_length=self.max_length,
                df=df,
                text_col=section,
                text2_col=other_section,
            )
            # if they are unlabeled, then we don't want to shuffle because it would then be harder to subset out...
            dataloader = DataLoader(
                dataset=dataset, batch_size=self.batch_size, shuffle=False
            )
        return dataloader

    def save_checkpoint(self, model, optimizer, step, section):
        filename = section + "_" + "selftrain_step" + str(step) + ".pt"
        torch.save(
            {
                "selftrain_step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join(self.logdir, filename),
        )
        
    def resume_from_checkpoint(self, step, section):
        model = transformers.DistilBertForSequenceClassification.from_pretrained(
            self.model1,
            num_labels=self.num_classes,  # The number of output labels
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )
        
        optimizer = AdamW(model.parameters(), lr=self.learning_rate)
        
        filename = section + "_" + "selftrain_step" + str(step) + ".pt"
        
        checkpoint = torch.load(os.path.join(self.logdir, filename))
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.cuda()
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return model, optimizer

    def train(self, model, val_dataloader, df, section, optimizer, epochs, step):
        # load dataset
        dataloader = self.load_dataset(section, df.reset_index(drop=True))

        # finetune model
        total_steps = len(dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        # Put the model into training mode.
        model.train()
        
        last_epoch_accuracy = 0

        for epoch_i in range(self.num_epochs):
            # For each batch of training data...
            for i, batch in enumerate(dataloader):
                # Unpack this training batch from our dataloader.
                b_input_ids = batch["ids"].cuda()
                b_input_mask = batch["mask"].cuda()
                b_labels = batch["target"].cuda()

                model.zero_grad()

                result = model(
                    b_input_ids,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                    return_dict=True,
                )

                loss = result.loss

                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                # Update the learning rate.
                scheduler.step()

                if section == "impressions":
                    self.impressions_global_step += 1
                    title = "Loss_" + section
                    self.writer.add_scalar(
                        title, loss, self.impressions_global_step
                    )  # this has to be an integer....

                else:
                    self.findings_global_step += 1
                    title = "Loss_" + section
                    self.writer.add_scalar(title, loss, self.findings_global_step)
            
            
            accuracy = self.eval(model=model,dataloader=val_dataloader, section=section, step=-1, type_df="val", save=False)
            if accuracy < last_epoch_accuracy:
                print("The best epoch is", epoch_i-1, "for step", step, "of cotraining")
                break
             
            self.save_checkpoint(model, optimizer, step, section)
            last_epoch_accuracy = accuracy
        
        model, optimizer = self.resume_from_checkpoint(step, section)

        return model, optimizer
    

    def get_conf_data(
        self,
        model_inf,
        dataloader_inf,
        coverage,
        section,
        other_section,
        step,
    ):
        softmax = nn.Softmax(dim=-1)

        model_inf.eval()
        # the df created must have the same y column and also
        df_inf = pd.DataFrame(
            columns=["File Name", "probability_inf", other_section, self.target]
        )
        for i, batch in enumerate(dataloader_inf):
            # Unpack this training batch from our dataloader.
            b_input_ids = batch["ids"].cuda()
            b_input_mask = batch["mask"].cuda()
            b_file_name = list(batch["file"][0])
            # for unlabeled data, there is an extra column where the other view is also returned
            b_other_view = list(batch["other_view"][0])
            
            with torch.no_grad():
                result_inf = model_inf(
                    b_input_ids,
                    attention_mask=b_input_mask,
                    return_dict=True,
                )

            # change logits into probability scores and then save them with the file ids...this way we can
            # then use these file ids to output the unlabeled df with the confident scores
            probabilities_inf = (
                np.max(softmax(result_inf.logits).detach().cpu().numpy(), axis=1)
                .flatten()
                .tolist()
            )

            # we only need the labels for the inference model...because we will pick the ones that are somewhat contrastrive to them...
            labels_inf = (
                np.argmax(softmax(result_inf.logits).detach().cpu().numpy(), axis=1)
                .flatten()
                .tolist()
            )

            new_batch = pd.DataFrame(
                {
                    "File Name": b_file_name,
                    "probability_inf": probabilities_inf,
                    other_section: b_other_view,
                    self.target: labels_inf,
                }
            )

            df_inf = pd.concat([df_inf, new_batch])

       
        df = df_inf
        print("Shape of merged inf and trn is", df.shape)
        
         # min_size = 10000
        # for i in range(self.num_classes):
        #     nrows = df[df[self.target] == i].shape[0]
        #     if nrows < min_size:
        #         min_size = nrows
        
        # sample_size = math.ceil(min_size*coverage)
        # conf_df = pd.DataFrame(columns=list(df.columns.values))
        # print("Column Names:",list(df.columns.values))
        # for i in range(self.num_classes):
        #     df = df[df[self.target] == i].sort_values(by=["probability_inf"], ascending=False)
        #     print("Predicted", str(i), "size:",df.shape[0])
        #     subset_df = df.iloc[:sample_size,:]
        #     conf_df = pd.concat([conf_df, subset_df])
        
        
        # return conf_df[["File Name", other_section, self.target]]


        if self.num_classes == 2:
            event_df = df[df[self.target] == 1].sort_values(by=["probability_inf"], ascending=False)
            nevent_df = df[df[self.target] == 0].sort_values(by=["probability_inf"], ascending=False)

            print("Predicted events size:", event_df.shape[0])
            print("Predicted non-event size:", nevent_df.shape[0])

            sample_size = math.ceil(min(event_df.shape[0], nevent_df.shape[0]) * coverage)
            conf_event_df = event_df.iloc[:sample_size,:]
            conf_nevent_df = nevent_df.iloc[:sample_size,:]

            conf_df = pd.concat([conf_event_df, conf_nevent_df])
            self.writer.add_scalar("Conf_data_len" + "-" + section, conf_df.shape[0], step)

            # record down percentage of positive values
            # event_fract = top_df[top_df[self.target] == 1].shape[0] / top_df.shape[0]
            return conf_df[["File Name", other_section, self.target]]
        else:
            event_df = df[df[self.target] == 1].sort_values(by=["probability_inf"], ascending=False)
            nevent_df = df[df[self.target] == 0].sort_values(by=["probability_inf"], ascending=False)
            likely_df = df[df[self.target] == 2].sort_values(by=["probability_inf"], ascending=False)

            print("Predicted events size:", event_df.shape[0])
            print("Predicted non-event size:", nevent_df.shape[0])
            print("Predicted likely size:", likely_df.shape[0])

            sample_size = math.ceil(min(event_df.shape[0], nevent_df.shape[0], likely_df.shape[0]) * coverage)
            conf_event_df = event_df.iloc[:sample_size,:]
            conf_nevent_df = nevent_df.iloc[:sample_size,:]
            conf_likely_df = likely_df.iloc[:sample_size,:]

            conf_df = pd.concat([conf_event_df, conf_nevent_df, conf_likely_df])
            self.writer.add_scalar("Conf_data_len" + "-" + section, conf_df.shape[0], step)

            # record down percentage of positive values
            # event_fract = top_df[top_df[self.target] == 1].shape[0] / top_df.shape[0]
            return conf_df[["File Name", other_section, self.target]]


    def eval(self, model, dataloader, section, step, type_df, save=True):

        # softmax function that we need for metric calculations:
        softmax = nn.Softmax(dim=-1)

        # store the prob, preds and labels
        probs = []
        preds = []
        labels = []

        model.eval()
        for i, batch in enumerate(dataloader):
            b_input_ids = batch["ids"].cuda()
            b_input_mask = batch["mask"].cuda()
            b_labels = batch["target"].cuda()

            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                result = model(
                    b_input_ids,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                    return_dict=True,
                )

            logits = result.logits

            # Transform probabilities and labels to a list so that we can use them to calculate auroc, auprc, other metrics
            probabilities = (
                softmax(logits).detach().cpu().numpy()
            )  # only keep the probabilities for the positive scores
            probabilities = probabilities[:, 1].flatten().tolist()
            predictions = (
                np.argmax(logits.detach().cpu().numpy(), axis=1).flatten().tolist()
            )
            label_ids = b_labels.cpu().numpy().flatten().tolist()

            probs += probabilities  # for event only though
            preds += predictions
            labels += label_ids

        # print("Results for", section, "view")

        if save==True:

            print("Results for", section, "view", "-", type_df)
            # metric calculations:
            if self.num_classes == 2:
                auroc = roc_auc_score(labels, probs)
                # auprc:
                precision, recall, thresholds = precision_recall_curve(labels, probs)
                auprc = auc(recall, precision)
                print("AUROC: {0:.6f} | AUPRC : {1:.6f} ".format(auroc, auprc))
                self.writer.add_scalar("AUROC" + "_" + section, auroc, step)
                self.writer.add_scalar("AUPRC" + "_" + section, auprc, step)
            else:
                auroc = 0
                auprc = 0

            # accuracy:
            accuracy = np.sum(np.array(preds) == np.array(labels)) / len(labels)
            print("Accuracy : {0:.6f} ".format(accuracy))
            self.writer.add_scalar("Accuracy" + "_" + section+ "_" + type_df, accuracy, step)

            results = pd.DataFrame(
                {
                    "selftrain step": step,
                    "val_test": type_df,
                    "section": section,
                    "Accuracy": accuracy,
                    "AUROC": auroc,
                    "AUPRC": auprc,
                },
                index=[0],
            )
            
            

            if self.results is None:
                columns = [
                    "selftrain step",
                    "val_test",
                    "section",
                    "Accuracy",
                    "AUROC",
                    "AUPRC",
                ]
                self.results = pd.DataFrame(columns=columns)

            self.results = pd.concat([self.results, results])
            # save results in a csv:
            self.results.to_csv(os.path.join(self.logdir, "results.csv"), index=False)
        
        accuracy = np.sum(np.array(preds) == np.array(labels)) / len(labels)
        print("Accuracy:", accuracy)
        return accuracy

    def ensemble_eval(
        self, model1, model2, dataloader1, dataloader2, section1, section2, step, type_df
    ):
        # softmax function that we need for metric calculations:
        softmax = nn.Softmax(dim=-1)

        # store the prob, preds and labels
        view1_probs = np.zeros((0, self.num_classes))
        view1_labels = []
        view1_file_names = []

        model1.eval()
        for i, batch in enumerate(dataloader1):
            b_input_ids = batch["ids"].cuda()
            b_input_mask = batch["mask"].cuda()
            b_labels = batch["target"].cuda()
            b_file_name = list(batch["file"][0])

            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                result = model1(
                    b_input_ids,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                    return_dict=True,
                )

            logits = result.logits

            # Transform probabilities and labels to a list so that we can use them to calculate auroc, auprc, other metrics
            probabilities = (
                softmax(logits).detach().cpu().numpy()
            )  # only keep the probabilities for the positive scores
            # probabilities = probabilities[:, 1].flatten().tolist()
            label_ids = b_labels.cpu().numpy().flatten().tolist()

            view1_probs = np.concatenate((view1_probs, probabilities), axis=0)
            view1_labels += label_ids
            view1_file_names += b_file_name

        view2_probs = np.zeros((0, self.num_classes))
        view2_labels = []
        view2_file_names = []

        model2.eval()
        for i, batch in enumerate(dataloader2):
            b_input_ids = batch["ids"].cuda()
            b_input_mask = batch["mask"].cuda()
            b_labels = batch["target"].cuda()
            b_file_name = list(batch["file"][0])

            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                result = model2(
                    b_input_ids,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                    return_dict=True,
                )

            logits = result.logits

            # Transform probabilities and labels to a list so that we can use them to calculate auroc, auprc, other metrics
            probabilities = softmax(logits).detach().cpu().numpy()
            label_ids = b_labels.cpu().numpy().flatten().tolist()

            view2_probs = np.concatenate((view2_probs, probabilities), axis=0)
            view2_labels += label_ids
            view2_file_names += b_file_name

        # first check if the view1 and view2 labels are the same:
        if view1_file_names == view2_file_names:
            print("The files names are the same")
        else:
            print("Check code")

        # combine the probabilities together
        # since they are for the 1 label, then if avg_prob < 0.5 then we choose 0, else we choose 1 for the pred_label
        probabilities = np.add(view1_probs, view2_probs)
        avg_probs = np.divide(probabilities, 2)
        pred_labels = np.argmax(avg_probs, axis=1)

        print("Results for ensemble")

        # metric calculations:
        if self.num_classes == 2:
            event_probs = avg_probs[:, 1]
            # auroc
            auroc = roc_auc_score(view1_labels, event_probs)
            # auprc:
            precision, recall, thresholds = precision_recall_curve(
                view1_labels, event_probs
            )
            auprc = auc(recall, precision)
            print("AUROC: {0:.6f} | AUPRC : {1:.6f} ".format(auroc, auprc))
        else:
            auroc = 0
            auprc = 0 

        # accuracy:
        accuracy = np.sum(np.array(pred_labels) == np.array(view1_labels)) / len(
            view1_labels
        )
        print("Accuracy : {0: .6f} ".format(accuracy))
        self.writer.add_scalar("Accuracy" + "_Ensemble"+"_"+type_df, accuracy, step)

        results = pd.DataFrame(
            {
                "selftrain step": step,
                "val_test" : type_df,
                "section": "Ensemble",
                "Accuracy": accuracy,
                "AUROC": auroc,
                "AUPRC": auprc,
            },
            index=[0],
        )

        self.results = pd.concat([self.results, results])
        # save results in a csv:
        self.results.to_csv(os.path.join(self.logdir, "results.csv"), index=False)
        return accuracy

    def selftrain(self, seed):
        # assign dataloaders first:
        view1_test_dataloader = self.load_dataset(
            section="findings",
            df=self.test_df.reset_index(drop=True),
            labeled=True,
            shuffle=False,
        )

        view2_test_dataloader = self.load_dataset(
            section="impressions",
            df=self.test_df.reset_index(drop=True),
            labeled=True,
            shuffle=False,
        )

        view1_val_dataloader = self.load_dataset(
            section="findings",
            df=self.val_df.reset_index(drop=True),
            labeled=True,
            shuffle=False,
        )

        view2_val_dataloader = self.load_dataset(
            section="impressions",
            df=self.val_df.reset_index(drop=True),
            labeled=True,
            shuffle=False,
        )

        # Findings model
        model1 = transformers.DistilBertForSequenceClassification.from_pretrained(
            self.model1,
            num_labels=self.num_classes,  # The number of output labels
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        ).cuda()

        # Impressions model
        model2 = transformers.DistilBertForSequenceClassification.from_pretrained(
            self.model2,
            num_labels=self.num_classes,  # The number of output labels
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        ).cuda()

        # Set optimizers for each view:
        view1_optimizer = AdamW(model1.parameters(), lr=self.learning_rate)
        view2_optimizer = AdamW(model2.parameters(), lr=self.learning_rate)

        print("Finetuning the models")
        print("Shape of train_df:", self.train_df.shape)
        
        # Finetune each view
        view1_finetuned_model, _ = self.train(
            model=model1,
            df=self.train_df,
            val_dataloader = view1_val_dataloader,
            section="findings",
            optimizer=view1_optimizer,
            epochs=self.num_epochs,
            step=0,
        )
        view2_finetuned_model, _ = self.train(
            model=model2,
            df=self.train_df,
            val_dataloader = view2_val_dataloader,
            section="impressions",
            optimizer=view2_optimizer,
            epochs=self.num_epochs,
            step=0,
        )


        # evaluation of every cotrain step:
        self.eval(model=view1_finetuned_model,dataloader=view1_test_dataloader, section="findings", step=0, type_df="test")
        self.eval(model=view2_finetuned_model,dataloader=view2_test_dataloader, section="impressions", step=0, type_df="test")

        

        print("Cotraining begins")

        view1_df = self.train_df[["File Name", "findings", self.target]]
        view2_df = self.train_df[["File Name", "impressions", self.target]]

        view1_unlabeled_dataloader = self.load_dataset(
            section="findings",
            df=self.unlabeled_df.reset_index(drop=True),
            labeled=False,
            other_section="findings",
        )

        view2_unlabeled_dataloader = self.load_dataset(
            section="impressions",
            df=self.unlabeled_df.reset_index(drop=True),
            labeled=False,
            other_section="impressions",
        )

        # set the finetuned models as the different views first:
        view1_model = view1_finetuned_model
        view2_model = view2_finetuned_model

        finetuned_ensemble_acc=self.ensemble_eval(
                model1=view1_model,
                model2=view2_model,
                dataloader1=view1_test_dataloader,
                dataloader2=view2_test_dataloader,
                section1="findings",
                section2="impressions",
                step=0,
                type_df = "test",
            )
        
        best_ensemble_val_accuracy = 0
        test_accuracy = 0
        findings_acc = 0
        impressions_acc = 0

        # Co-train each view:
        for i in range(1, self.selftrain_steps + 1):
            print("cotrain step", i)
            coverage = self.init_coverage
            view1_labeled = self.get_conf_data(
                model_inf=view1_model,
                dataloader_inf=view1_unlabeled_dataloader,
                section="findings",
                coverage=coverage,
                other_section="findings",
                step=i,
            )
            
            view1_train = pd.concat([view1_df, view1_labeled])
            print("Shape of view1_train:", view1_train.shape)
            # print("Shape of view1_train:", view2_labeled.shape)
            # print(view2_labeled.columns)

            view1_model, _ = self.train(
                model=model1,
                df=view1_train,
                val_dataloader=view1_val_dataloader,
                section="findings",
                optimizer=view1_optimizer,
                step=i,
                epochs=self.num_epochs,
            )
            
            view2_labeled = self.get_conf_data(
                model_inf=view2_model,
                dataloader_inf=view2_unlabeled_dataloader,
                section="impressions",
                coverage=coverage,
                other_section="impressions",
                step=i,
            )
            
            # append it to the train df
            # create a subset of the train df
            view2_train = pd.concat([view2_df, view2_labeled])
            print("Shape of view2_train:", view2_train.shape)
            print("Shape of view1_labeled", view2_labeled.shape)
            # print(view1_labeled.columns)

            view2_model, _ = self.train(
                model=model2,
                df=view2_train,
                val_dataloader=view2_val_dataloader,
                section="impressions",
                optimizer=view2_optimizer,
                step=i,
                epochs=self.num_epochs,
            )

            print("The {0:.0f} selftrain step".format(i))

            # evaluation of every cotrain step:

            # findings
            findings_val_acc = self.eval(model=view1_model,dataloader=view1_val_dataloader, section="findings", step=i, type_df="val")
            findings_test_acc = self.eval(model=view1_model, dataloader=view1_test_dataloader, section="findings", step=i, type_df="test")
            
            # impressions
            impressions_val_acc = self.eval(model=view2_model,dataloader=view2_val_dataloader, section="impressions", step=i, type_df="val")
            impressions_test_acc = self.eval(model=view2_model, dataloader=view2_test_dataloader, section="impressions", step=i, type_df="test")
            
            # ensemble
            ensemble_val_acc = self.ensemble_eval(
                model1=view1_model,
                model2=view2_model,
                dataloader1=view1_val_dataloader,
                dataloader2=view2_val_dataloader,
                section1="findings",
                section2="impressions",
                step=i,
                type_df = "val",
            )

            ensemble_test_acc = self.ensemble_eval(
                model1=view1_model,
                model2=view2_model,
                dataloader1=view1_test_dataloader,
                dataloader2=view2_test_dataloader,
                section1="findings",
                section2="impressions",
                step=i,
                type_df = "test",
            )
            
            if ensemble_val_acc > best_ensemble_val_accuracy:
                best_ensemble_val_accuracy = ensemble_val_acc
                test_accuracy = ensemble_test_acc
                selftrain_step = i
                findings_acc = findings_test_acc
                impressions_acc = impressions_test_acc
            
        # record save the file:
        best_results = pd.DataFrame({
            "Seed": seed,
            "Best ensemble val accuracy": best_ensemble_val_accuracy,
            "Ensemble test accuracy" : test_accuracy,
            "Findings test accuracy": findings_acc,
            "Impressions test accuracy": impressions_acc,
            "Selftrain step": selftrain_step,
        }, index=[0])
        
        results_df = pd.read_csv(self.all_results_path)
        
        results_df = pd.concat([results_df, best_results])
        results_df.to_csv(self.all_results_path)
        
        return


if __name__ == "__main__":
    # parse some arguments that are needed
    argparser = argparse.ArgumentParser()

    # Data Processing
    argparser.add_argument(
        "--annotations-file",
        type=str,
        help="The csv that contains human annotations for radiology reports",
        default="../labels.csv",
    )  # change
    argparser.add_argument(
        "--mets-file", type=str, help="The csv that matches file names to mets info"
    )
    argparser.add_argument(
        "--labeled-folder-path",
        type=str,
        help="Folder path to where the labeled reports are stored",
    )
    argparser.add_argument(
        "--labeled-pickle",
        type=str,
        help="Stores the cleaned labeled data. If store-data is activated then this argument sets where the new pkl file will be stored",
    )
    argparser.add_argument(
        "--store-data",
        action="store_true",
        help="If true, cleans the data again and creates a new pickle file",
    )
    argparser.add_argument(
        "--unlabeled-folder-path",
        type=str,
        help="Folder path to where the unlabeled reports are stored",
    )
    argparser.add_argument(
        "--unlabeled-pickle",
        type=str,
        help="Stores the cleaned unlabeled data. If store-data is activated then this argument sets where the new pkl file will be stored",
    )
    argparser.add_argument(
        "--max-length",
        type=int,
        help="The max number of tokens per sequence",
        default=512,
    )
    argparser.add_argument("--random-seed", type=int, default=0)
    argparser.add_argument("--test-split", type=float, default=0.4)

    # Training
    argparser.add_argument("--batch-size", type=int, default=16)
    argparser.add_argument("--num_classes", type=int, default=2)
    argparser.add_argument("--learning-rate", type=float, default=5e-5)
    argparser.add_argument("--num-epochs", type=int, default=3)
    argparser.add_argument("--target", type=str, default="mass_label")

    # Cotraining
    argparser.add_argument("--selftrain-steps", type=int, default=5)
    argparser.add_argument("--init-coverage", type=float, default=0.25)
    argparser.add_argument("--unlabeled-size", type=int, default=10000, help="1k,3k,5k,10k")

    # Path to tensorboard and csv of results:
    argparser.add_argument(
        "--logdir",
        type=str,
        default="log/",
        help="Path to save results to",
    )
    
    # Overall results:
    argparser.add_argument(
        "--all-results-path", 
        type=str, 
        default="selftrain.csv", 
        help="Path to save best selftrain step results"
    )

    args = argparser.parse_args()

    # Set device:
    device = torch.device("cuda")

    # Determine whether to create a new pickle file or to use existing one:
    if args.store_data:
        # clean annotations and match annotations with cleaned radiology reports
        organize_labeled_data(
            annotations_file=args.annotations_file,
            met_info_file=args.mets_file,
            report_path=args.labeled_folder_path,
        ).to_pickle(args.labeled_pickle)
        # clean unlabeled data put them into a pickle file
        organize_unlabeled_data(report_path=args.unlabeled_folder_path).to_pickle(
            args.unlabeled_pickle
        )

        labeled_df = pd.read_pickle(args.labeled_pickle)
        unlabeled_df = pd.read_pickle(args.unlabeled_pickle)
        print(unlabeled_df.shape)
        # unlabeled_df = unlabeled_df.iloc[:args.unlabeled_size,:]

    else:
        labeled_df = pd.read_pickle(args.labeled_pickle)
        unlabeled_df = pd.read_pickle(args.unlabeled_pickle)
        unlabeled_df = unlabeled_df.iloc[:args.unlabeled_size,:]
        print("shape of unlabeled", unlabeled_df.shape)


    torch.cuda.empty_cache()
    
    kf = KFold(n_splits=5, random_state=0, shuffle=True)
    
    run = 0
    
    for train_index, test_index in kf.split(labeled_df):
        print("Run" + str(run) + "for random seed", +  args.random_seed)
        df_tr_va, df_test = labeled_df.iloc[train_index], labeled_df.iloc[test_index]
        train_df, val_df = train_test_split(df_tr_va, test_size=0.25, random_state=0)
        test_df = labeled_df.iloc[test_index]
        
        print("Train DF shape:", train_df.shape)
        print("Val DF shape:", val_df.shape)
        print("Test DF shape:", test_df.shape)
        print(args.num_epochs)
        
        # Set the seed value all over the place to make this reproducible.
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

        os.makedirs(args.logdir+"_"+str(run), exist_ok=True)
        
        # run 10k, 20k, 30k 
        # initialize co-training model:
        method = SelfTrain(
            model1="distilbert-base-cased",
            model2="distilbert-base-cased",
            logdir=args.logdir+"_"+str(run),
            train_df=train_df,
            test_df=test_df,
            val_df = val_df,
            unlabeled_df=unlabeled_df.iloc[:args.unlabeled_size,:],
            max_length=args.max_length,
            batch_size=args.batch_size,
            num_classes=args.num_classes,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            target=args.target,
            selftrain_steps=args.selftrain_steps,
            init_coverage=args.init_coverage,
            all_results_path=args.all_results_path,
        )

        method.selftrain(seed=args.random_seed)
        
        run += 1

