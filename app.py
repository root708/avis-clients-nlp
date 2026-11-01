"""Application Streamlit — Spacy et Hugging Face.

Saisir un avis client : l'application affiche l'avis anonymisé, la prédiction des deux modèles
(TF-IDF + régression logistique, mini-BERT) et les mots qui ont le plus pesé pour TF-IDF.

Lancer :  streamlit run app.py
Fichiers attendus à côté de app.py (produits par le notebook) :
    tfidf_avis.joblib, config_app.json, mini_bert_avis/
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DOSSIER = Path(__file__).parent

# ----------------------------------------------------------------------------- anonymisation
RE_CMD = re.compile(r"\bCMD-\d+\b")
RE_TEL = re.compile(r"(?<!\d)(?:\+?221[ .\-]?)?7[05678](?:[ .\-]?\d){7}(?!\d)")
RE_MONTANT = re.compile(
    r"(?<![\w+.])(?:\d{1,3}(?:[ .\u00a0\u202f]\d{3})+|\d+)\s?(?:FCFA|CFA|francs?|F)\b", re.I)
NEG = {"ne", "pas", "jamais", "rien", "aucun", "plus", "sans"}


def reperer(texte, nlp):
    """Renvoie [(debut, fin, etiquette)] ; les regex passent avant spaCy, sans chevauchement."""
    spans = []
    libre = lambda a, b: not any(a < e and s < b for s, e, _ in spans)
    for etiquette, rx in [("COMMANDE", RE_CMD), ("TEL", RE_TEL), ("MONTANT", RE_MONTANT)]:
        for m in rx.finditer(texte):
            if libre(m.start(), m.end()):
                spans.append((m.start(), m.end(), etiquette))
    for ent in nlp(texte).ents:
        if ent.label_ == "PER" and libre(ent.start_char, ent.end_char):
            spans.append((ent.start_char, ent.end_char, "NOM"))
    return sorted(spans)


def anonymiser(texte, nlp):
    for debut, fin, etiquette in reversed(reperer(texte, nlp)):
        texte = texte[:debut] + f"[{etiquette}]" + texte[fin:]
    return texte


# ----------------------------------------------------------------------------- modèles
@st.cache_resource(show_spinner="Chargement de spaCy…")
def charger_spacy():
    import spacy
    try:
        return spacy.load("fr_core_news_md")
    except OSError:
        from spacy.cli import download
        download("fr_core_news_md")
        return spacy.load("fr_core_news_md")


@st.cache_resource(show_spinner="Chargement du modèle TF-IDF…")
def charger_tfidf():
    import joblib
    chemin, cfg = DOSSIER / "tfidf_avis.joblib", DOSSIER / "config_app.json"
    if not (chemin.exists() and cfg.exists()):
        return None, None
    return joblib.load(chemin), json.loads(cfg.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Chargement du mini-BERT…")
def charger_bert():
    from transformers import BertForSequenceClassification, PreTrainedTokenizerFast
    dossier = DOSSIER / "mini_bert_avis"
    if not dossier.exists():
        return None, None
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(dossier))
    return tokenizer, BertForSequenceClassification.from_pretrained(str(dossier)).eval()


def preparer(texte, mode, nlp):
    """Même prétraitement que dans le notebook pour la variante TF-IDF retenue."""
    if mode == "brut":
        return texte
    garder_neg = mode == "lem_garde_neg"
    with nlp.select_pipes(disable=["parser", "ner"]):
        doc = nlp(texte)
    return " ".join(t.lemma_.lower() for t in doc if not t.is_punct and not t.is_space
                    and (not t.is_stop or (garder_neg and (t.lemma_.lower() in NEG or t.lower_ in NEG))))


def proba_tfidf(pipeline, texte_prepare):
    return float(pipeline.predict_proba([texte_prepare])[0, 1])


def contributions(pipeline, texte_prepare, k=8):
    """Poids (coefficient x valeur TF-IDF) des n-grammes présents dans l'avis ; >0 pousse vers « positif »."""
    vec, clf = pipeline[0], pipeline[1]
    X = vec.transform([texte_prepare])
    idx = X.nonzero()[1]
    if len(idx) == 0:
        return pd.DataFrame(columns=["poids"])
    noms = np.asarray(vec.get_feature_names_out())[idx]
    poids = X[0, idx].toarray().ravel() * clf.coef_[0][idx]
    df = pd.DataFrame({"poids": poids}, index=noms)
    return df.reindex(df["poids"].abs().sort_values(ascending=False).index).head(k)


def proba_bert(tokenizer, modele, texte):
    import torch
    enc = tokenizer([texte], truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        return float(torch.softmax(modele(**enc).logits, -1)[0, 1])


def verdict(p):
    return ("positif" if p > 0.5 else "négatif"), (p if p > 0.5 else 1 - p)


# ----------------------------------------------------------------------------- interface
EXEMPLES = {
    "— choisir un exemple —": "",
    "Avis classique": "Franchement, le chauffeur était courtois et le taxi est arrivé vite. Je recommande !",
    "Négation (piège)": "Ce n'était pas mauvais du tout.",
    "Avis nuancé": "Le taxi était sale mais le trajet a été rapide.",
    "Wolof": "Le mafé, neexul.",
    "Ironie (piège)": "Bravo pour les deux heures de retard.",
    "Avec données personnelles": "Bonjour, je suis Moussa Diop. Ma commande CMD-12345 coûte 12 500 FCFA, rappelez-moi au 77 123 45 67.",
}


def main():
    st.set_page_config(page_title="Ce que disent les clients", page_icon="💬", layout="centered")
    st.title("💬 Ce que disent les clients")
    st.caption("· Spacy et Hugging Face · Données fictives (restaurants, livraison, transport, boutiques)")

    pipeline, cfg = charger_tfidf()
    tokenizer, bert = charger_bert()
    if pipeline is None and bert is None:
        st.error("Modèles introuvables. Exécutez d'abord le notebook : il crée `tfidf_avis.joblib`, "
                 "`config_app.json` et le dossier `mini_bert_avis/` à placer à côté de `app.py`.")
        st.stop()
    nlp = charger_spacy()

    choix = st.selectbox("Exemples", list(EXEMPLES))
    texte = st.text_area("Avis ou message à analyser", value=EXEMPLES[choix], height=120,
                         placeholder="Écrivez un avis en français (le wolof est possible, mais mal compris).")
    if not st.button("Analyser", type="primary") or not texte.strip():
        st.stop()

    st.subheader("1. Avis anonymisé")
    st.code(anonymiser(texte, nlp), language=None)
    st.caption("Noms, téléphones, montants et numéros de commande sont masqués par [NOM], [TEL], [MONTANT], [COMMANDE].")

    st.subheader("2. Prédictions")
    col1, col2 = st.columns(2)
    verdicts = {}
    if pipeline is not None:
        prep = preparer(texte, cfg["mode_tfidf"], nlp)
        p = proba_tfidf(pipeline, prep)
        verdicts["TF-IDF"] = verdict(p)[0]
        col1.metric("TF-IDF + régression logistique", verdict(p)[0], f"confiance {verdict(p)[1]:.0%}", delta_color="off")
    if bert is not None:
        p_b = proba_bert(tokenizer, bert, texte)
        verdicts["mini-BERT"] = verdict(p_b)[0]
        col2.metric("Mini-BERT (2 couches)", verdict(p_b)[0], f"confiance {verdict(p_b)[1]:.0%}", delta_color="off")
    if len(set(verdicts.values())) > 1:
        st.warning("Les deux modèles ne sont pas d'accord : c'est souvent le signe d'un avis difficile (négation, nuance, ironie, wolof).")

    if pipeline is not None:
        st.subheader("3. Mots qui ont le plus pesé (TF-IDF)")
        df = contributions(pipeline, prep)
        if df.empty:
            st.info("Aucun mot de l'avis n'est connu du modèle.")
        else:
            st.bar_chart(df["poids"], horizontal=True)
            st.caption("Barre positive : pousse vers « positif » ; négative : vers « négatif ».")

    with st.expander("Limites de l'outil"):
        st.markdown(
            "- Modèles entraînés sur un corpus **fictif** et très régulier : un bon score ne prouve pas une vraie compréhension.\n"
            "- Négations inédites, ironie, comparaisons et wolof sont des points faibles.\n"
            "- Ne pas utiliser ces prédictions pour classer ou sanctionner de vrais commerces.")
        if cfg:
            st.caption(f"Variante TF-IDF : {cfg.get('variante')} · F1 macro test : TF-IDF {cfg.get('f1_test_tfidf')}, "
                       f"mini-BERT {cfg.get('f1_test_bert')}")


if __name__ == "__main__":
    main()
