"""
Author: Armin Berger
Date: 10.01.23

Practice Project which seeks to use Pytorch and FluidML to classify SMS text
messages as spam (1) or not (0).
"""
import pandas as pd
import os
import nltk

# nltk.download()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import fluidml
from fluidml.common import Task
from fluidml.swarm import Swarm
from fluidml.flow import Flow, GridTaskSpec, TaskSpec
from fluidml.visualization import visualize_graph_interactive
from fluidml.visualization import visualize_graph_in_console
from fluid_helper import (
    MyLocalFileStore,
    TaskResource,
    configure_logging,
    get_balanced_devices,
)

from preprocess import PreProcessData
from featurize import FeaturizeText
from model import TrainModel


def main():

    # create all task specifications
    preprocess_text = TaskSpec(
        task=PreProcessData,
        config={"file_name": "data/train.csv"},
        publishes=["preprocessed_text"],
    )

    featurize_text = TaskSpec(
        task=FeaturizeText,
        publishes=["featurized_text"],
    )

    train_model = TaskSpec(task=TrainModel, publishes=["model"])

    # create dependencies between tasks
    featurize_text.requires(preprocess_text)
    train_model.requires(featurize_text)

    # define all tasks
    tasks = [preprocess_text, featurize_text, train_model]

    # get base directory to store experiment results
    base_dir = "Desktop/code/ml_test/output"

    # set other run configuration variables
    num_workers = 2
    use_cuda = False
    cuda_ids = None
    run_name = None

    # create list of resources
    devices = get_balanced_devices(
        count=num_workers, use_cuda=use_cuda, cuda_ids=cuda_ids
    )
    resources = [TaskResource(device=devices[i]) for i in range(num_workers)]

    # create local file storage used for versioning
    results_store = MyLocalFileStore(base_dir=base_dir, run_name=run_name)

    # run the pipeline
    flow = Flow(tasks=tasks)

    # # graph visualization of our tasks
    # visualize_graph_in_console(
    #     graph=flow.task_spec_graph, use_unicode=False
    # )

    flow.run(num_workers=num_workers, resources=resources, project_name="ml-test")


if __name__ == "__main__":

    main()
