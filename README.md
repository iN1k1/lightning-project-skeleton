# ⚡️ Lightning Skeleton Library

The Lightning Skeleton Library is a template repository designed to help you quickly set up a new machine learning
project using PyTorch Lightning. This repository provides a structured and scalable framework for your deep learning
experiments, including model training, evaluation, and data handling.

## 💡 Idea Behind the Repository

## 🚀 Getting Started

### 📥 Clone this Repository

```shell
git clone git@github.com:iN1k1/lightning-project-skeleton.git
```

### 🌐 Create Your GitHub Repository

Create a new repository on GitHub, for example: `git@github.com:iN1k1/new-amazing-project` .

### 🤖 Run the Automation Script

To automate the whole renaming process, just run the provided script and folow the instructions:

```shell
./rename_repo.sh
```

## 🛠️ Example Usage

To run an example training script, use the following command:

```shell
PYTHONPATH=./src python scripts/train.py --config ./configs/example/dummy_example.py
```

## 🔧 Adapting the Repository for Other Projects

You need to **work on the configuration files**: The `configs` folder contains configuration files that define the
model, loss,
optimizer, scheduler, and dataset settings. These configurations are used to instantiate components dynamically.

### Configuration Files

The `configs` folder contains example configuration files. Here is an example structure:

```plaintext
configs/
├── example/
│   ├── dummy_data.py
│   ├── dummy_example.py
│   ├── dummy_model.py
├── default_data_runtime.py
├── default_model_runtime.py
```

### Example Configuration (`dummy_example.py`)

The configuration files use a dictionary format to define various components of the machine learning pipeline. Each
component is specified using a `target` key, which indicates the class or function to be instantiated, and a `params`
key, which provides the parameters to be passed to the target.

Within the `dummy_model.py` you find a the main `model` config dictionary containig the following keys:
- `modelconfig`: Defines the model architecture. The `target` is the model class (e.g., `torchvision.models.resnet18`),
  and
  `params` include any initialization parameters (e.g., `pretrained`, `num_classes`).

- `lossconfig`: Specifies the loss function. The `target` is the loss class (e.g., `torch.nn.CrossEntropyLoss`), and
  `params` include any parameters required by the loss function.

- `optimizer`: Configures the optimizer. The `target` is the optimizer class (e.g., `torch.optim.Adam`), and `params`
  include hyperparameters like learning rate (`lr`).

- `scheduler`: Sets up the learning rate scheduler. The `target` is the scheduler class (e.g.,
  `torch.optim.lr_scheduler.StepLR`), and `params` include scheduler-specific settings (e.g., `step_size`, `gamma`).

Within the `dummy_data.py` you find a similar structure using the `data` config dictionary containig the following keys:

- `train` and `val`: Define the datasets to be used. The `target` is the dataset class (e.g., `lightning_project_skeleton.data.example.ExampleDataset`),
  and `params` include dataset-specific options (e.g., `phase`, `transform`).

The `_base_` key is used to include common configurations shared across multiple configuration files. It allows for
reusability and modularity by referencing a base configuration file and extending or overriding its settings.

### Instantiating Components

The core functionality of the configuration files is to instantiate components dynamically based on the provided
dictionary.
This is achieved using the `instantiate_from_config` function, which takes a configuration dictionary and returns the
instantiated object.

For instance the `instantiate_from_config` function is used in the `BaseTaskModel` class to instantiate the model and
the loss function as:

```python
from lightning_project_skeleton.build.from_config import instantiate_from_config


class BaseTaskModel(LightningBaseModel):
    def __init__(self, modelconfig, lossconfig):
        super().__init__(**kwargs)
        self.model = instantiate_from_config(modelconfig)
        self.loss = instantiate_from_config(lossconfig)
```

## 🧑‍💻 Development

### 📏 Code Guidelines

- We use the `black` formatter to format code and enforce style. You are encouraged to do the same.
- Use strong types wherever possible to ensure code clarity and maintainability.

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.



