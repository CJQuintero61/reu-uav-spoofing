# reu-uav-spoofing
This repo contains the code related to the Summer 2026 Research Experience for Undergraduates (REU).

Included are machine learning (ML) models used to detect UAV GPS spoofing signals.

Authors:
- Christian Quintero
- Lilyanna Yang

## To Run
First create a virtual environment (venv)
1. python -m venv .venv
2. .venv/Scripts/activate
3. pip install -r requirements.txt

Then, setup your python venv as your kernel for running the notebooks
1. open a notebook
2. select choose a kernel
3. select python environments
4. select create python environments
5. select enter interpreter path
6. choose ./.venv/Scripts/python.exe to select your created venv as your kernel

Then choose your venv as your python kernel when running the notebook

## References
The dataset used can be found at https://dx.doi.org/10.21227/00dg-0d12

Jason Whelan, Thanigajan Sangarapillai, Omar Minawi, Abdulaziz Almehmadi, Khalil El-Khatib, "UAV Attack Dataset", IEEE Dataport, February 26, 2020, doi:10.21227/00dg-0d12 