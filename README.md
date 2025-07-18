# Thesis_PredictingNewsOutdatedness_Logistic_Decay

This repository contains the complete implementation and documentation for my master's thesis with the topic: "Predicting News Outdatedness Using a Probabilistic Logistic Decay Model"

The research question is “How can a probabilistic model to predict the outdatedness of a news data using logistic decay model be developed?”

The results can be found on W&B (see results at the bottom of README) and the thesis can be found in docs (see repository structure).

# Script usage:

The script is designed to be run in two different environments:

Local Machine: Used for the initial stages, including dataset generation and data annotation.

Google Colab: Used for a more intensive tasks such as: BERT fine-tuning, Hyperparameter tuning, Prediction, Fitting to the decay function

# Repository structure


# Training results
Rule-Based Validation: https://wandb.ai/spurwand-hwr-berlin/thesis_news_outdatedness_validation_rulebased?nw=nwuserspurwand

Hyperparam Tuning: https://wandb.ai/spurwand-hwr-berlin/thesis_finetune_bert_optuna_tuning_tpe?nw=nwuserspurwand

Model with best hyperparam: https://wandb.ai/spurwand-hwr-berlin/thesis_finetune_bert_optuna_bestparam?nw=nwuserspurwand

Decay function: https://wandb.ai/spurwand-hwr-berlin/thesis_decay_fit_evaluation_bestparam
