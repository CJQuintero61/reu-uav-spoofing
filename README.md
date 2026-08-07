# reu-uav-spoofing
This repo contains the code related to the Summer 2026 Research Experience for Undergraduates (REU).

Included are machine learning (ML) models used to detect UAV GPS spoofing signals.

Authors:
- Lilyanna Yang
- Christian Quintero


# To Run

Create a virtual environment

1. python -m venv .venv
2. .venv\Scripts\activate
3. pip install -r requirements.txt

Then, download the CSV file all_spoofed_30_flights.csv fom the data folder. The CSV is located in the zip file.
Place this csv file in src/main_handlers.

Next, navigate to src/main_handlers. In the __main__ function of client_models_by_run,
you can update what model you want to run by seeing the model key names located in model_helper/factories.py,
then you can change how many training/testing runs by updating the second parameter, and lastly you can choose what
parameter config you'd like for a particular model by looking at model_helper/parameter_dic.py and adding
a model's parameter config to their '_para' parameter list to test.


Lastly, from the root directory, run `python -m src.main_handlers.client_models_by_run`