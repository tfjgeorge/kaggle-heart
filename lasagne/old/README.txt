Salut!

Dans ce dossier se trouve tous les modèles codés avec Lasagne. Il y a trois fichiers et en somme, trois modèles.

Les fichiers/modèles sont:

- Independant_NN : Même modèle que Keras/Nachos avec, en plus, des batchs normalisation layers entres les layers
de convolution et les layers d'activation. Il a aussi été rajouté la norme correspondant à celle présente sur Kaggle.
Un réseau de neurones pour d'une part la prédiction du systole et d'autres part la prédiction du diastole sont 
nécessaires.

- Merged_NN : obtenu à partir du merge de deux Independant_NN. Un modèle pour la prédiction du diastole et du systole (en même temps). La
première couche est commune puis le réseau de neurones diverge en deux réseaux indépendant. La norme et le
batch normalisation sont inchangés par rapport à Independant_NN.

- Recurrent_NN : obtenu à partir de Merged_NN en ajoutant une couche récurrente après les couches de convolution. L'idée est de rendre les
neurones dépendant car le systole dépend probablement qqpart du diastole.

Rq : for space reason, the data folder has been emptied