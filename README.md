# Classification of items in Estonian Museums
Part of this kaggle challenge: https://www.kaggle.com/competitions/caps-in-museums-data-part-2

For collaboration: https://docs.google.com/document/d/1UM7QEFyb16v7Zd-cRcEMz8tcKqE91UEl0C2PXt6B85A/edit?usp=sharing

Column discovery: https://docs.google.com/spreadsheets/d/1E4Z6RELIxHR8ZOggO6AyRTGMs3EQOBxrMZo25ahPMXM/edit?usp=sharing

# Getting started
Embed the cloned project in a PyVenv https://docs.python.org/3/library/venv.html (or conda env if you prefer) and install the requiremnts listed in requirements.txt


# Executing
Mostly you will see Jupyter Notebooks. Especially during data preparation we went in baby steps therefore you will often notice something similar to:

data = dataset.copy
"do something with data"
data.to_csv('dataset.csv)

execute this block - to have the changes present in the current notebook you will have to read the files in again via "from setup_general import *" and you can proceed in the notebook from top to bottom.
