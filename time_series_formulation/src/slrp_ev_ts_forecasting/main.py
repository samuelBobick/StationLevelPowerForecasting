from default_parameters import TypeModelChoice
from run_one_model import run_one_model

model_choice: TypeModelChoice = "Basic_NN"

if __name__ == "__main__":
    run_one_model(model_choice=model_choice)
