import pandas as pd
import nltk
import streamlit as st

# ambil rule probabilistic
df = pd.read_csv("rules/probabilistic_rules.csv")

# tulis rule cfg
grammar_cfg = ""
for prod in df["production"]:
  grammar_cfg += f"{prod}\n"
grammar_cfg = nltk.CFG.fromstring(grammar_cfg)
grammar_cfg._start = nltk.Nonterminal("K")
parser_cfg = nltk.ChartParser(grammar_cfg)

# tulis kalimat
sentence = st.text_input("Enter your Balinese sentence: ")
button = st.button("Parse")

if button:
  sentence = sentence.lower().split()

  # hitung probabilistic dan memilih hasil parsing dengan prob tertinggi
  result = pd.DataFrame({"tree": list(parser_cfg.parse(sentence)), "probability": [1 for _ in range(len(list(parser_cfg.parse(sentence))))]})

  if len(result) == 0:
    st.info("This sentence haven't parsing result")
  for i, value_i in result.iterrows():
    for j in value_i["tree"].productions():
      for k, value_k in df.iterrows():
        if nltk.CFG.fromstring(value_k["production"]).productions()[0] == j:
          result["probability"][i] *= value_k["probability"]
          continue

  # merangking rules berdasarkan probability yang dihasilkan
  result = result.sort_values(by='probability', ascending=False).reset_index(drop=True)
  svg = result["tree"][0]._repr_svg_()
  st.write(svg, unsafe_allow_html=True)
  st.info(f"Score: {result["probability"][0]}")



