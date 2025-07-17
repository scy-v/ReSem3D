# ReSem3D: Refinable 3D Spatial Constraints via Fine-Grained Semantic Grounding for Generalizable Robotic Manipulation

<h3 align="center">
  <a href="https://resem3d.github.io/">[Project Page]</a>
</h3>

<p align="center">
  <img src="videos/task.gif" alt="task video">
</p>

---

## 🛠️ Environment

- **OS**: Ubuntu 20.04  
- **CUDA**: 12.2  
- **NVIDIA Driver**: 535.161.07  
- **Conda Environments**:  
  - `ReSem3D` (Client)  
  - `m3p2i-aip` (Server)  

---

## 🔧 Client Setup (`ReSem3D`)

### 1. Configure Conda Channels

```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
```

### 2. Create Conda Environment

```bash
conda create -n ReSem3D python=3.10 
conda activate ReSem3D
conda config --set channel_priority flexible
conda config --remove channels conda-forge
```

### 3. Install [OmniGibson](https://behavior.stanford.edu/omnigibson/getting_started/installation.html)

Install from source (editable mode), version `v1.1.0`:

```bash
git clone -b v1.1.0 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K/
```

Install compatible PyTorch version (check CUDA compatibility):

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```

Install OmniGibson:

```bash
pip install -e .
python -m omnigibson.install
pip install Pillow==9.4.0
```

> ⚠️ If you encounter errors related to `*-manylinux_2_31_*.whl not found`, rename all files in the ISAAC_SIM_PACKAGES folder from `linux_2_34` to `linux_2_31`.

### 4. Test Installation

```bash
cd <BEHAVIOR-1K_folder>
python -m omnigibson.examples.robots.robot_control_example --quickstart
```

### 5. Clone the ReSem3D Repository

```bash
git clone https://github.com/scy-v/ReSem3D.git
```

### 6. Install FastSAM and Download Weights

```bash
cd ReSem3D
git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
pip install -r requirements_client.txt
```

Download the [FastSAM model weights](https://drive.usercontent.google.com/download?id=1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv&export=download)  
Place the weights inside the `ReSem3D/weights/` directory.

---

## 🖥️ Server Setup (`m3p2i-aip`)

### 1. Clone the `m3p2i-aip` Repository and Create Environment

```bash
cd <ReSem3D_folder>
git clone https://github.com/tud-amr/m3p2i-aip.git
conda create -n m3p2i-aip python=3.8
conda activate m3p2i-aip
```

### 2. Install Isaac Gym

Follow the [prerequisites guide](https://github.com/tud-amr/m3p2i-aip/blob/master/thirdparty/README.md), download Isaac Gym from NVIDIA:[https://developer.nvidia.com/isaac-gym](https://developer.nvidia.com/isaac-gym)

Move and install Isaac Gym:

```bash
mv <Downloaded_Folder>/IsaacGym_Preview_4_Package <ReSem3D_folder>/m3p2i-aip/thirdparty/
cd <ReSem3D_folder>/m3p2i-aip/thirdparty/IsaacGym_Preview_4_Package/isaacgym/python
pip install -e .
```

### 3. Install `m3p2i-aip`

```bash
cd <ReSem3D_folder>/m3p2i-aip
pip install -e .
```

### 4. Install Additional Dependencies

```bash
cd <ReSem3D_folder>
pip install -r requirements_server.txt
```

---

## 🚀 Running the Demo

You need two terminals: one for the **server** and one for the **client**.

### 1. Start the Server

```bash
cd <ReSem3D_folder>
python mppi_server.py
```

### 2. Run the Client

```bash
cd <ReSem3D_folder>
python run.py [--load_cache] [--visualize]
```

- `--load_cache`: Load pre-generated GPT-4o cache  
- `--visualize`: Enable visual debugging

---

## 🙏 Acknowledgments

- **Simulation Environments**  
  The simulation environments are based on [OmniGibson](https://behavior.stanford.edu/omnigibson/getting_started/installation.html) and [Isaac Gym](https://developer.nvidia.com/isaac-gym).

- **Language Model Integration**  
  The extension of Language Model Programs (LMPs) is built upon [Voxposer](https://voxposer.github.io/) and [Code as Policies](https://code-as-policies.github.io/).

- **Motion Planning**  
  The Model Predictive Path Integral (MPPI) algorithm implemented on Isaac Gym is adopted from [m3p2i-aip](https://autonomousrobots.nl/paper_websites/m3p2i-aip).

- **Code Snippets Reference**  
  Some code segments come from the [ReKep](https://rekep-robot.github.io/) project.

