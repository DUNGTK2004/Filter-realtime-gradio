
## 🚀  Quickstart

```bash
# clone project
git clone https://github.com/DUNGTK2004/Filter-realtime-gradio.git
cd Filter-realtime-gradio

# [OPTIONAL] create conda environment
conda create -n myenv python=3.12.7
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

### Pipeline
1. Checkpoint for model landmark detection
Download the checkpoints [model.pth](https://drive.google.com/file/d/1bBLc2_Wz0eb7Fbw9szEL5kr83TKWcQl1/view?usp=sharing) at here and place it at filter/src/models/checkpoint
# run app
python -m filter.app

Finally access the localhost link in the terminal to start the app
