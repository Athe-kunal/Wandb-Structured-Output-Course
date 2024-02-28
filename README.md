## PROJECT MOTIVATION

This project involves the instructor library for structured output generation. This helps us to get the tickers, years and quarters that the question is talking about without asking the user explicitly. To setup the project

```python
python -m venv wandb-env
source wandb-env/bin/activate
```

Install the dependencies

```python
pip install -r requirements.txt
```

Place the `.env` file inside the `src` folder and root folder with your API keys

Run the chainlit file for the application

```python
chainlit run chainlit.py -w
```
