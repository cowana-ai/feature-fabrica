<h4 align="center">
    <img alt="Feature Fabrica logo" src="https://raw.githubusercontent.com/cowana-ai/feature-fabrica/main/media/current_logo.png" style="width: 100%;">
</h4>
<h2>
    <p align="center">
     ⚙️ The Framework to Simplify and Scale Feature Engineering ⚙️
    </p>
</h2>

<p align="center">
    <a href="https://colab.research.google.com/drive/1O9i-g3vmxyazwdadTVjgBlY1GFN4f7Xt?usp=sharing">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
    </a>
</p>

<p align="center">
    <img src="https://img.shields.io/pypi/v/feature-fabrica?style=flat-square" alt="PyPI version"/>
    <img src="https://img.shields.io/github/stars/cowana-ai/feature-fabrica?style=flat-square" alt="Stars"/>
    <img src="https://img.shields.io/github/issues/cowana-ai/feature-fabrica?style=flat-square" alt="Issues"/>
    <img src="https://img.shields.io/github/license/cowana-ai/feature-fabrica?style=flat-square" alt="License"/>
    <img src="https://img.shields.io/github/contributors/cowana-ai/feature-fabrica?style=flat-square" alt="Contributors"/>
    <img src="https://app.codacy.com/project/badge/Grade/5df9f22c8a2d49a08058bf8a660b086c" alt="Code Quality"/>

</p>

For **data scientists, ML engineers**, and **AI researchers** who want to simplify feature engineering, manage complex dependencies, and boost productivity.

______________________________________________________________________

## Introduction

**Feature Fabrica** is an open-source Python library designed to improve engineering practices and transparency in feature engineering. It allows users to define features declaratively using YAML, manage dependencies between features, and apply complex transformations in a scalable and convenient manner.

By providing a structured approach to feature engineering, Feature Fabrica aims to save time, reduce errors, and enhance the transparency and reproducibility of your machine learning workflows. Whether you're working on small projects or managing large-scale pipelines, **Feature Fabrica** is designed to meet your needs.

## **Key Features**

- **📝 Declarative Feature Definitions**: Define features, data types, and dependencies using a simple YAML configuration.
- **🔄 Transformations**: Apply custom transformations to raw features to derive new features.
- **🔗 Dependency Management**: Automatically handle dependencies between features.
- **✔️ Pydantic Validation**: Ensure data types and values conform to expected formats.
- **🛡️ Fail-Fast with Beartype**: Catch type-related errors instantly during development, ensuring your transformations are robust.
- **🚀 Scalability**: Designed to scale from small projects to large machine learning pipelines.
- **🔧 Hydra Integration**: Leverage Hydra for configuration management, enabling flexible and dynamic configuration of transformations.

______________________________________________________________________

## 🛠️ Quick Start

### Installation

To install **Feature Fabrica**, simply run:

```bash
pip install feature-fabrica
```

### **Defining Features in YAML**

Features are defined in a YAML file. See examples in `examples/` folder. Here’s an example:

```yaml
feature_a:
  description: "Raw feature A"
  data_type: "int32"
  group: "training"

feature_b:
  description: "Raw feature B"
  data_type: "float32"
  group: "training"
  transformation:
    scale_feature:
      _target_: ().scale(factor=2)

feature_c:
  description: "Derived feature C"
  data_type: "float32"
  group: "training_experiment"
  dependencies: ["feature_a", "feature_b"]
  transformation:
    solve:
      _target_: (feature_a + feature_b) / 2

feature_e:
  description: "Raw feature E"
  data_type: "int32"
  group: "draft"
  transformation:
    _target_: ().upper().lower().one_hot(categories=['apple', 'orange'])
```

### **Creating and Using Transformations**

You can define custom transformations by subclassing the Transformation class:

```python
from feature_fabrica.transform import Transformation


class MyCustomTransform(Transformation):
    _name_ = "my_custom_transform"

    def execute(self, data):
        return data * 2
```

```yaml
feature_a:
  description: "Raw feature A"
  data_type: "int32"
  group: "training"
  transformation:
    _target_: ().my_custom_transform()
```

### **Compiling and Executing Features**

To compile and execute features:

```python
import numpy as np
from feature_fabrica.core import FeatureManager

data = {
    "feature_a": np.array([10.0], dtype=np.float32),
    "feature_b": np.array([20.0], dtype=np.float32),
}
feature_manager = FeatureManager(
    config_path="../examples", config_name="basic_features"
)
results = feature_manager.compute_features(data)
print(results["feature_c"])  # 0.5 * (10 + 20) = 15.0
print(results.feature_c)  # 0.5 * (10 + 20) = 15.0
```

### Visualize Features and Dependencies

Track & trace Transformation Chains

```python
import numpy as np
from feature_fabrica.core import FeatureManager

data = {
    "feature_a": np.array([10.0], dtype=np.float32),
    "feature_b": np.array([20.0], dtype=np.float32),
}
feature_manager = FeatureManager(
    config_path="../examples", config_name="basic_features"
)
results = feature_manager.compute_features(data)
print(feature_manager.features.feature_c.get_transformation_chain())
# Transformation Chain: (Transformation: sum_fn, Value: 30.0 Time taken: 9.5367431640625e-07 seconds) -> (Transformation: scale_feature, Value: 15.0, Time taken:  9.5367431640625e-07 seconds)
```

Visualize Dependencies

```python
from feature_fabrica.core import FeatureManager

feature_manager = FeatureManager(
    config_path="../examples", config_name="basic_features"
)
feature_manager.get_visual_dependency_graph()
```

![image.png](media/example.png)

# Contributing to Feature Fabrica

First, thank you for taking the time to contribute! 🎉 Contributions are essential to making **Feature Fabrica** a better library, and we truly appreciate your involvement.

The following is a set of guidelines for contributing to **Feature Fabrica**, including reporting bugs, adding new features, and improving documentation.

## Roadmap

- NLP support
- Embeddings support
- Simplify UI
- Better visualizations/reports

## Table of Contents

1. [How to Contribute](#how-to-contribute)
2. [Reporting Bugs](#reporting-bugs)
3. [Suggesting Enhancements](#suggesting-enhancements)

______________________________________________________________________

## How to Contribute

### Fork and Clone the Repo

1. **Fork** the repository to your own GitHub account by clicking the "Fork" button at the top of the page.

2. **Clone** your fork locally:

   ```bash
   git clone https://github.com/your-username/feature-fabrica.git
   cd feature-fabrica
   ```

3. Set the original repository as a remote:

   ```bash
   git remote add upstream https://github.com/cowana-ai/feature-fabrica.git
   ```

4. Before creating a new branch, ensure your `main` branch is up-to-date:

   ```bash
   git checkout main
   git pull upstream main
   ```

### Create a Branch

1. Create a new branch for your feature or bug fix:

   ```bash
   git checkout -b feature/my-new-feature
   ```

2. Make your changes in this new branch.

______________________________________________________________________

## Reporting Bugs

If you discover a bug in **Feature Fabrica**, please [open an issue](https://github.com/cowana-ai/feature-fabrica/issues) on GitHub. Before submitting your report, please check if an issue already exists to avoid duplicates. Include the following details in your report:

- A clear and concise description of the bug.
- Steps to reproduce the issue.
- Expected behavior vs. actual behavior.
- If applicable, screenshots or code snippets.

______________________________________________________________________

## Suggesting Enhancements

We welcome suggestions to improve **Feature Fabrica**. Feel free to [open an issue](https://github.com/cowana-ai/feature-fabrica/issues) describing the enhancement. Please be as detailed as possible in describing:

- The feature you'd like to see.
- The reason it would be beneficial.
- Any potential drawbacks.

______________________________________________________________________
