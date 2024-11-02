# Building a Multi-Task NLP Model for Real-World Applications

MultiNLP is a multi-task learning project focused on classifying Yelp business reviews into broader categories and analyzing their sentiment. It utilizes a transformer-based model for text classification and sentiment analysis.

## Project Overview

This project uses multi-task learning with a transformer-based model to:
1. Classify reviews into broad categories such as "Restaurants," "Beauty & Spas," "Automotive," etc.
2. Predict the sentiment of each review as positive, neutral, or negative.

The model is trained on the Yelp dataset, specifically the review and business files, and supports text preprocessing, category mapping, and multi-task training.

## Setup

### Prerequisites

Ensure that you have the following installed:
- Python 3.8 or higher (if running without Docker)
- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) (for containerized setup)

### Cloning the Repository

Clone this repository to your local machine:
```bash
git clone https://github.com/Tilak559/MultiNLP.git
cd MultiNLP
```

### Required Files

Download the Yelp dataset JSON files and place them in a folder named `yelp_dataset` within the project directory.

Required files:
- `yelp_academic_dataset_review.json`
- `yelp_academic_dataset_business.json`

The structure should look like this:

```
MultiNLP/
├── src/
│   ├── main.py
│   ├── model.py
│   └── utils.py
├── yelp_dataset/
│   ├── yelp_business.json
│   └── yelp_review.json
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── output.log
```


## Running the Project

### Option 1: Running Without Docker

1. **Setting up the Virtual Environment**

    If you prefer to run the project locally, first set up a virtual environment:

    ```bash
    python -m venv .venv
    ```

2. **Activating the Virtual Environment**

    - On MacOS/Linux:
      ```bash
      source .venv/bin/activate
      ```
    - On Windows:
      ```bash
      .venv\Scripts\activate
      ```

3. **Installing Dependencies**

    Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

4. **Running the Application**

    Once the dependencies are installed, you can start the data preparation and training process by running:
    ```bash
    python src/main.py
    ```

### Option 2: Running with Docker

1. **Building the Docker Image**

    Use Docker Compose to build the Docker image:
    ```bash
    docker-compose build
    ```

2. **Starting the Docker Container**

    Start the container using Docker Compose:
    ```bash
    docker-compose up -d
    ```

    This will start the container and run the `main.py` script.

3. **Viewing Logs**

    To view the logs of the running container in real-time, use:
    ```bash
    docker logs -f multinlp_container
    ```

4. **Stopping the Docker Container**

    To stop the container, run:
    ```bash
    docker-compose down
    ```

## Training the Model

The `main.py` script is used for training the model with a limited number of iterations to observe initial accuracy. Training logs and accuracy metrics are output in real-time.

### Training Parameters

- **Iterations**: The script is set to run for 100 iterations by default. You can adjust this in the `train` function in `main.py`.
- **Batch Size**: Currently set to 1. Adjust it in the `train_dataloader` initialization in `main` to fit your system's memory.
- **Learning Rate**: Set to `1e-5` in `main.py`. You may experiment with other values for optimal performance.

## Configuring Hyperparameters

Hyperparameters such as learning rate, batch size, and the number of training iterations can be customized in the `main.py` script. For example:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Adjust learning rate here
```

## Blog

[Building a Multi-Task NLP Model for Real-World Applications](https://medium.com/@tilak559/building-a-multi-task-nlp-model-for-real-world-applications-95d6b5dd1d17)

## Acknowledgments

- **Transformers Library**: [Hugging Face Transformers](https://github.com/huggingface/transformers) for pre-trained models and tokenizers.
- **Yelp Dataset**: [Yelp Open Dataset](https://www.yelp.com/dataset) for review and business data.

