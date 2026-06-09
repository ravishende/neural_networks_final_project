Final Project for COMPSCI 274P: Neural Networks

# Setup

## Clone the project

`git clone https://github.com/ravishende/neural_networks_final_project.git`
`cd neural_networks_final_project`

## Create a virtual environment with required libraries

```
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

# Running the code

1. `cd src`
2. Open a desired jupyter notebook and run its cells in order
   - Make sure you have a gpu accessible (e.g. on google colab)

## Running on a remote server (ssh)

1. ssh into your remote server
2. cd into the repository (run the setup section commands if it's not yet created)
3. run `jupyter notebook <DESIRED_NOTEBOOK_NAME>.ipynb --no-browser --port=8080`
4. open a new terminal locally
5. In the local terminal, run `ssh -L 8080:localhost:8080 <REMOTE_USER>@<REMOTE_HOST>`
6. open a web browser and go to `http://localhost:8080/`
