# TLED: Training-Based Approximate Layer Exploration in DNNs with Efficient Multipliers

TLED (Training-Based Approximate Layer Exploration in DNNs with Efficient Multipliers) is an innovative approach aimed at enhancing the performance and efficiency of Deep Neural Networks (DNNs) through the integration of training-based approximate layer exploration and efficient multiplier designs. This repository contains the essential components to facilitate this approach, including PyTorch code for TLED layers, Verilog models for Partial Product Column Tailoring (PPCT) multipliers, and scripts for synthesis with the ASAP 7nm process library.

## Contents

- **TLED-layers**: Contains the PyTorch implementation of TLED layers, along with CUDA and C++ files for acceleration purposes.
- **PPCT**: Features Verilog models of PPCT multipliers that support the omission of 2 to 11 columns for optimized performance.
- **DC**: Includes scripts for synthesizing multipliers and accelerators using the [Arizona State Predictive PDK (ASAP) 7nm process library](https://github.com/The-OpenROAD-Project/asap7) within Synopsys Design Compiler.

## TLED-layers Overview

- `tled_layers.py`: This Python file provides classes for TLED layers, including both convolutional and dense layers.
- `GPU-accelerate.cu`: A CUDA file that accelerates the functions used in `tled_layers.py`.
- `tled_layers.cpp`: Acts as a bridge between PyTorch and CUDA for seamless integration.
- `tled_layers.h`: Header files for C++ implementations.

## PPCT Multipliers

The PPCT folder contains Verilog files for PPCT multipliers, designed to enhance the efficiency of DNNs by tailoring partial product columns. These models range from eliminating 2 columns up to 11 columns, providing a versatile set of tools for DNN optimization.

## Synthesis Scripts

The `DC` directory hosts scripts essential for the synthesis of multipliers and accelerators, specifically tailored for the ASAP 7nm process library. These scripts enable users to leverage the advanced features of the Synopsys Design Compiler for optimal hardware realization.

## Getting Started

To use the components in this repository, follow these general steps:

1. **Installation**: Ensure you have PyTorch1.11.0+cu102, CUDA, and Synopsys Design Compiler installed on your system.
2. **TLED-layers Setup**: Integrate the TLED layers into your DNN models by importing `tled_layers.py` and ensuring CUDA acceleration with `GPU-accelerate.cu`.
3. **PPCT Multipliers**: Incorporate the PPCT multiplier Verilog models into your hardware design to enhance efficiency.
4. **Synthesis with DC**: Use the provided scripts in the `DC` directory to synthesize your designs with the ASAP 7nm process library for optimized performance.

