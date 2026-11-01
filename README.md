# 💬 Ce que disent les clients — spaCy et Hugging Face

TP6 de **Python avancé** (L3 Intelligence Artificielle, ISM École d'ingénieurs, séance 6/8) — projet en binôme.

Analyse d'avis clients **fictifs** (restaurants, livraison, transport, boutiques, de Dakar à Ziguinchor) : anonymisation des données personnelles, analyse linguistique, classement positif/négatif avec TF-IDF et avec un petit transformer entraîné depuis zéro, puis étude de ce que les modèles comprennent réellement.

## Contenu du dépôt

| Fichier | Rôle |
| --- | --- |
| `TP6_Spacy_et_Hugging_Face.ipynb` | Notebook complet (parties 1 à 6) |
| `app.py` | Application Streamlit : avis anonymisé, prédiction des deux modèles, mots les plus influents |
| `requirements.txt` | Dépendances Python |
| `messages_anonymises.csv` | Messages du service client après anonymisation (livrable de la partie 2) |
| `tokenizer_avis.json` | Tokenizer WordPiece entraîné sur le corpus |
| `tfidf_avis.joblib`, `config_app.json`, `mini_bert_avis/` | Modèles sauvegardés par le notebook, utilisés par l'application |

Les fichiers de données `avis_clients.csv` et `messages_service_client.csv` sont fournis par l'enseignant et sont des données générées pour le cours.

## Ce que fait le projet

1. **Audit du corpus** : taille, équilibre des classes, doublons, textes à deux étiquettes, longueurs, wolof, avis nuancés (« mais »).
2. **Anonymisation** (spaCy + regex) : noms de personnes, numéros de téléphone sénégalais, montants (FCFA, CFA, F, francs) et numéros de commande, remplacés par `[NOM]`, `[TEL]`, `[MONTANT]`, `[COMMANDE]`. Précision et rappel mesurés à la main sur 30 messages.
3. **Analyse linguistique** (spaCy) : lexique de la satisfaction par domaine, sujets des phrases négatives (dépendances), structure des avis avec « mais », place du wolof.
4. **Référence TF-IDF** + régression logistique : 7 variantes comparées sur la validation (dont l'effet du retrait de « ne » et « pas »), test évalué une seule fois.
5. **Mini-BERT** (Hugging Face `tokenizers` + `transformers`) : tokenizer WordPiece, 2 couches, époques choisies sur la validation.
6. **Ce que comprennent les modèles** : typologie des erreurs, étiquettes douteuses, 10 phrases écrites à la main, équité entre domaines.

Protocole : doublons retirés avant le découpage 60/20/20 (`random_state=0`, stratifié), choix sur la validation, test utilisé une seule fois, métrique F1 macro.

## Résultats

À compléter après l'exécution du notebook (voir aussi `config_app.json`).

| Modèle | F1 macro (test) |
| --- | --- |
| TF-IDF + régression logistique | … |
| Mini-BERT 2 couches | … |

## Installation

Python **3.10 à 3.12** recommandé.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download fr_core_news_md
```

## Utilisation

1. Placer `avis_clients.csv` et `messages_service_client.csv` à côté du notebook.
2. Ouvrir `TP6_Spacy_et_Hugging_Face.ipynb` et lancer « Tout exécuter ». Le notebook produit les modèles et les fichiers d'export. L'entraînement du mini-BERT est plus rapide sur GPU (Google Colab par exemple).
3. Lancer l'application :

```bash
streamlit run app.py
```

## Limites

- Corpus fictif, très régulier : un score élevé ne prouve pas une vraie compréhension de la langue.
- Négations inédites, ironie, comparaisons et wolof restent des points faibles des deux modèles.
- Les prédictions ne doivent pas servir à classer ou à sanctionner de vrais commerces.
- Le mini-BERT n'est pas pré-entraîné : sur un petit corpus, il n'a aucune raison de dépasser TF-IDF.

## Auteurs

Papa Arona SENE et NDIAYE ……, L3 IA, ISM École d'ingénieurs. Enseignant : Elias BADJI.
