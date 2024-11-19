# ai_project_aes

## Evaluation Results

| Model            | MSE       | Accuracy | F1 Score | Quadratic Kappa Score |
|------------------|-----------|----------|----------|------------------------|
| ALBERT only      | 205.85851 | 0.55237  | 0.43735  | 0.74535                |
| ALBERT + GloVe   | 205.72769 | 0.50224  | 0.39994  | 0.74897                |
| ALBERT + FastText| 246.73769 | 0.66697  | 0.63933  | 0.70059                |


## Sample Essay Scores

| Model             | Bad Essay Score | Bad Essay Quality | Good Essay Score | Good Essay Quality |
|-------------------|-----------------|-------------------|------------------|---------------------|
| ALBERT only       | 89.62041        | Medium           | 89.62041        | Medium             |
| ALBERT + GloVe    | 88.81711        | High             | 88.81711        | High               |
| ALBERT + FastText | 81.81429        | High             | 81.81429        | High               |


export PATH=$PATH:/home/salsabila.pranida/.conda/envs/ai_aes/bin


distilbert/distilbert-base-uncased https://huggingface.co/distilbert/distilbert-base-uncased
MBZUAI/MobiLlama-1B-Chat https://huggingface.co/MBZUAI/MobiLlama-1B-Chat
google/electra-small-discriminator https://huggingface.co/google/electra-small-discriminator
microsoft/MiniLM-L12-H384-uncased https://huggingface.co/microsoft/MiniLM-L12-H384-uncased
google/mobilebert-uncased https://huggingface.co/google/mobilebert-uncased

Predict the Overall Score:
You would use a regression model (like the one already defined) to predict the overall normalized_score.
Predict the Essay Type:
Use a classifier to predict the essay_type (i.e., argumentative, source-dependent, or narrative) based on input features or embeddings.
Predict Specific Attributes Based on Type:
Depending on the predicted essay_type, assess relevant attributes. For example, argumentative essays could focus on content, organization, and language, while narrative essays may emphasize narrativity and style.