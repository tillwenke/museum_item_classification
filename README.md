# Classification of items in Estonian Museums
Part of this [kaggle challenge](https://www.kaggle.com/competitions/caps-in-museums-data-part-2); [original data](https://opendata.muis.ee/)

Baseline: 100 tree RF: 30 % validation set [accuracy on kaggle test set: private: 0.901 public: 0.90266]
                               precision    recall  f1-score   support

                      ajakiri       1.00      0.71      0.83        31
                      ajaleht       0.42      0.35      0.38        23
                        album       0.25      0.14      0.18         7
          arheoloogiline leid       0.98      1.00      0.99       259
             aukiri/auaadress       1.00      0.38      0.55         8
                  diapositiiv       1.00      0.94      0.97        17
           digitaalne kujutis       0.98      1.00      0.99        44
                     dokument       0.58      0.70      0.63       125
                          ehe       1.00      0.50      0.67         6
                         foto       0.94      0.98      0.96      1114
              foto, postkaart       1.00      0.24      0.38        17
                 fotomaterjal       1.00      0.73      0.84        11
                 fotonegatiiv       0.98      1.00      0.99       664
   fotonegatiiv, fotonegatiiv       0.00      0.00      0.00         8
                     graafika       0.83      0.97      0.89        69
                helisalvestis       1.00      1.00      1.00        17
                    joonistus       1.00      0.53      0.69        17
                     kalender       1.00      0.81      0.90        16
                   karikatuur       1.00      0.71      0.83        14
                kaustik/vihik       0.00      0.00      0.00         1
                         kava       0.78      0.84      0.81       232
         kavand/joonis/eskiis       0.98      0.99      0.99       234
                         kiri       0.62      0.75      0.68       198
              kiri, postkaart       0.50      0.67      0.57         3
                        kleit       0.45      1.00      0.62         5
                         kott       1.00      0.25      0.40         4
                        kutse       1.00      0.29      0.44        35
                     käsikiri       0.84      0.76      0.80       144
käsikiri, laul/ vokaalmuusika       0.79      0.95      0.86        58
        käsikiri, muusikateos       0.40      0.10      0.16        20
             käsikiri, noodid       1.00      1.00      1.00         2
                  laegas/karp       0.50      0.50      0.50         6
                   lina/linik       1.00      0.83      0.91         6
                    lint/pael       0.00      0.00      0.00         7
                         maal       0.83      0.71      0.77         7
                        medal       1.00      0.90      0.95        31
            muusikainstrument       1.00      0.91      0.95        11
                      märkmed       0.00      0.00      0.00         5
                         münt       0.98      1.00      0.99        56
                       noodid       0.84      0.93      0.88        71
                         nukk       0.00      0.00      0.00         1
                     nõu/anum       1.00      0.80      0.89        10
                        paber       0.57      0.50      0.53         8
                       pakend       1.00      0.25      0.40         4
                       pitsat       1.00      0.75      0.86         4
         pitser/templijäljend       0.99      1.00      0.99        66
                       plakat       0.99      1.00      0.99       221
                    postkaart       0.50      0.20      0.28        51
                       raamat       0.94      0.97      0.95       154
                    silt/märk       0.47      0.75      0.58        12
                    skulptuur       0.00      0.00      0.00         3
                    telegramm       0.50      0.45      0.48        22
                    trükinoot       1.00      0.29      0.44         7
                    tunnistus       0.00      0.00      0.00         1
                  väiketrükis       0.83      0.61      0.70        33

                     accuracy                           0.89      4200
                    macro avg       0.73      0.61      0.64      4200
                 weighted avg       0.89      0.89      0.89      4200

In [Column Discovery](https://docs.google.com/spreadsheets/d/1E4Z6RELIxHR8ZOggO6AyRTGMs3EQOBxrMZo25ahPMXM/edit?usp=sharing) our findings and resulted feature engineering tasks per feature are captured.

# Getting started
Embed the cloned project in a [PyVenv](https://docs.python.org/3/library/venv.html) (or conda env if you prefer) and install the requiremnts listed in requirements.txt

Due to large files you have to get trained models and text embeddings from [here](https://drive.google.com/drive/folders/1ZKOynMoLZd0R_0Q8kB-AwELzFwo50atb?usp=sharing) and paste them respectively in models/ and data/text_embeddings

For getting a quick overview check out the respective folder.

# Executing
Mostly you will see Jupyter Notebooks. Especially during data preparation we went in baby steps therefore you will often notice something similar to:

```
data = dataset.copy
"do something with data"
data.to_csv('dataset.csv)
```

execute this block - to have the changes present in the current notebook you will have to read the files in again via `from setup_general import *` and you can proceed in the notebook from top to bottom.

We set settings so that Notebook outputs are fully displayed for investigation purposes. But this makes it necessary to limit what you want to see when displaying variables containing large amount of data, so for example rather do `prep.head()`than `prep`.



