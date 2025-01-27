# Create a virtual environment (Linux/Mac)
python -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate

(ensure using correct interpreter)

install depedencies inside venv

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

install NLTK stock words & tokenizers from script

```bash
python NLTKscript.py
```

May need to install lxml html clean

```bash
pip install lxml[html_clean]
```
or explicitly 

```bash
pip install lxml_html_clean
```

after installing run streamlit app
```bash
streamlit run app.py
```




Fake news articles can be found here:
https://www.bbc.com/news/topics/cjxv13v27dyt

Example 1:
https://www.bbc.com/news/articles/cy890gpqw1po


![Image](https://github.com/user-attachments/assets/048a48af-f0a8-45ef-b9cf-a2add0fdbe1c)

Example 2 real news:
https://www.cnn.com/2025/01/26/politics/colombia-tariffs-trump-deportation-flights/index.html

![Image](https://github.com/user-attachments/assets/13b8286a-077d-4709-af6a-ff5f991a05f7)

FakeNewsDetector/ <br/>
│ <br/>
├── app.py               # Streamlit app (main interface)  <br/>
├── train.py             # Script to train the model  <br/>
├── evaluate.py          # Evaluate the trained model  <br/>
├── model.py             # Model architecture or pipeline definition  <br/>
├── utils.py             # Helper functions (e.g., preprocessing, loading data)  <br/>
├── requirements.txt     # Dependencies (streamlit, pandas, transformers, etc.)  <br/>
├── data/                # Data directory  <br/>
│   ├── Fake.csv  <br/>
│   └── True.csv  <br/>
└── models/              # Directory to save trained models  <br/>
    └── fake_news_model.pkl  <br/>


dataset  <br/>
https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets  <br/>
