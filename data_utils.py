import pandas as pd
import numpy as np
import os
import re
import argparse
import transformers


# define a function that creates a wholistic label for the different classes
def mass_label(row):
    if row["Mass"] == "no":
        return 0
    elif row["Mass"] == "yes":
        return 1


def aggressive_label(row):
    if row["Aggressive"] == "yes":
        return 1
    elif row["Aggressive"] == "no" or row["Aggressive"] == "not applicable":
        return 0
    else:  # else condition is for when the label is possible/likely
        return 2


# Cleans the text in the report a bit for better processing
# Could be more thoroughly cleaned afterwards...
def clean_txt(text):
    # deletes extra spacing both between words and also new lines
    text = re.sub("\s{1,}", " ", text)
    text = text.replace("*", "").replace("_", "")

    # delete information regarding dictation and verification:
    if text.find("Edited") != -1:
        text = text[: text.find("Edited")]
    if text.find("Dictat") != -1:
        text = text[: text.find("Dictat")]
    if text.find("Electronically") != -1:
        text = text[: text.find("Electronically")]
    return text


# Cleans the raw annotations
def clean_label(label):
    return label.strip().lower()


# Determines start (start index) of impressions section if there is one:
# to-do: how to get the end of the word...not the beginning?
def impressions_ind(report):
    # impressions section is sometimes called different words, we must check if any of these words exist in the text and the smallest index of it
    loi = [
        report.lower().find("impression"),
        report.lower().find("opinion"),
        report.lower().find("summary"),
        report.lower().find("conclusion"),
        report.lower().find("interpretation"),
    ]
    if len(set(loi)) == 1:
        return -1
    else:
        loi = [i for i in loi if i != -1]
        return min(loi)


def get_sect_start_ind(report, heading_ind):
    start_ind = report.find(" ", heading_ind)
    if start_ind != -1:
        return start_ind
    return heading_ind


# Returns the findings section in the report:
def findings_sect(report):
    start = report.lower().find("finding")
    if start == -1:
        return "N/A"
    end = impressions_ind(report) if impressions_ind(report) > -1 else len(report)
    return report[get_sect_start_ind(report, start) : end]


# Returns the impressions section in the report:
def impressions_sect(report):
    start = impressions_ind(report)
    if start == -1:
        return "N/A"
    return report[get_sect_start_ind(report, start) :]


# load data into dataloader + tokenize
tokenizer = transformers.DistilBertTokenizer.from_pretrained("distilbert-base-cased")


# Returns the number of tokens a section has:
def n_tokens(section):
    return len(tokenizer.encode(section))


# Returns a categorical variable telling us how many sections a report has:
def report_sections(findings, impressions):
    f = len(findings)
    i = len(impressions)

    if f == i == 3:
        return "None"
    elif f == 3 and i > 3:
        return "Impressions"
    elif i == 3 and f > 3:
        return "Findings"
    elif i > 3 and f > 3:
        return "Both"
    return "Check"
