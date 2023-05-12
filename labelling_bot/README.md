# Labelling chat bot for UWG experts team

## Purpose
With this bot UWG experts team can send images for sorting samples from their mobile phoses directly to the storage. This labelled data will be used to train the garbage classifier model.

## How to use
Just open Telegram and find @nowaste_com_ua_labelling_bot. Send your images to the bot and it will save them to the storage with the corresponding labels.

## How to deploy
Just merge new version to main branch and it will be automatically deployed to the server.

Note! Use `config.json` to add new labels to the bot. If you want to add new labels to dataset - please add new labels to the end of the labels list. In this case you only can add new labels with new unique key. If you want to start to collect new dataset from scratch - please change `root_folder` key in `config.json` to the new one.