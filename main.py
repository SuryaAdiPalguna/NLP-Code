import pandas as pd
import nltk
import streamlit as st

# ambil rule probabilistic
df = pd.read_csv("rules/probabilistic_rules.csv")

# tulis rule pcfg
grammar_pcfg = ""
for prod, prob in zip(df["production"], df["probability"]):
  grammar_pcfg += f"{prod} [{prob}]\n"
grammar_pcfg = nltk.PCFG.fromstring(grammar_pcfg)
grammar_pcfg._start = nltk.Nonterminal("K")
parser_pcfg = nltk.ViterbiParser(grammar_pcfg)

st.set_page_config(page_title="Constituency Parsing", layout="centered")
st.title("MESIN PARSING KALIMAT BAHASA BALI")

# tulis kalimat
sentence = st.text_input("Enter your Balinese sentence: ")
button = st.button("Parse")

if button:
  sentence = sentence.lower().split()

  try:
    for tree in parser_pcfg.parse(sentence):
      svg = tree._repr_svg_()
      st.write(svg, unsafe_allow_html=True)
      st.info(f"Score: {tree.prob()}")
    if len(list(parser_pcfg.parse(sentence))) == 0:
      st.warning("This sentence haven't parsing result")
  except Exception as e:
    st.error(e)
