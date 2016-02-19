## m7x3DCNN_alternatetx

- **Spécificité modèle** : On alterne les convolutions dans le sens de l'image et dans le sens du temps
- **Lancé** :  19/2/16
- **Notes** :  Le data augmentation a changé au milieu, il y avait une erreur dans le multiply de randomdownscale
- **Commit**:  https://github.com/tfjgeorge/kaggle-heart/commit/02f89b74c6aea053c0383453dbd3eb059fc80393
- **Score 20 epochs**: 

```
Training status:
         batch_interrupt_received: False
         epoch_interrupt_received: False
         epoch_started: False
         epochs_done: 20
         iterations_done: 6361
         received_first_batch: True
         resumed_from: None
         training_started: True
Log records from the iteration 6361:
         loss: 64.4487838745
         time_read_data_this_epoch: 0.70716547966
         time_read_data_total: 14.3917677402
         time_train_this_epoch: 489.799297094
         time_train_total: 10002.5152304
         training_finish_requested: True
         training_finished: True
         valid_crps: 0.0956550911069
         valid_loss: 68.2937469482
```
