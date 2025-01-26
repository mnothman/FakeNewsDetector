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
