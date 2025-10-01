# Station-Level Power Forecasting and Joint Price and Power Optimization Algorithms 

## Setup

- To install the necessary packages, navigate to `time_series_formulation` and run `pip install -e .`.
- For dataset access, email ecal@berkeley.edu.
- `simulations/general_simulator.ipynb` contains infrastructure to test each optimizer with a Monte Carlo simulation and is a good starting point for understanding the codebase.
## Code 

Below we provide a brief description of each subdirectory. 

- `data`: stores utility functions used for data cleaning and feature engineering.
- `peak_power_formulation`: a very early iteration of this project; we are 99% it will not be useful for future work.
- `simulations`: contains various optimization algorithms for station-level joint price and power optimization algorithms. Each optimizer is named with the format ________Simulator.py. Every optimizer is a child of BaselineSimulator.py. `simulations/general_simulator.ipynb` contains infrastructure to test each optimizer with a Monte Carlo simulation.
- `time_series_formulation`: contains all logic for station-level power forecasting. Run time_series_formulation/src/slrp_ev_ts_forecasting/main.py to train, test, and validate a forecasting algorithm on the SLRP-EV dataset.

## Citation

If you use this repository or ideas from it, please cite the following work: 

Thibaud Cambronne, Samuel Bobick, Wente Zeng, and Scott Moura. "Joint Price and Power MPC for Peak Power Reduction at Workplace EV Charging Stations." *arXiv preprint* arXiv:2507.12703 [eess.SY], 2025. https://doi.org/10.48550/arXiv.2507.12703 


## Data 

Data used for this project is collected from SLRP-EV, a human-in-the-loop EV charging experiment conducted on the UC Berkeley campus. For dataset access, email ecal@berkeley.edu. For more information, see the following works:

Teng Zeng, Sangjae Bae, Bertrand Travacca, and Scott J. Moura. "Inducing human behavior to maximize operation performance at PEV charging station." *IEEE Transactions on Smart Grid*, vol. 12, no. 4, pp. 3353–3363, IEEE, 2021. https://doi.org/10.1109/TSG.2021.3067072

Hassan Obeid, Ayşe Tuğba Öztürk, Wente Zeng, and Scott J. Moura. "Learning and optimizing charging behavior at PEV charging stations: Randomized pricing experiments, and joint power and price optimization." *Applied Energy*, vol. 351, 121862, Elsevier, 2023. https://doi.org/10.1016/j.apenergy.2023.121862

Ayşe Tuğba Öztürk, Hassan Obeid, Teng Zeng, Wente Zeng, and Scott J. Moura. "Joint price and power optimization experiment for workplace charging stations." *Sustainable Cities and Society*, 106784, Elsevier, 2025. https://doi.org/10.1016/j.scs.2025.106784.

