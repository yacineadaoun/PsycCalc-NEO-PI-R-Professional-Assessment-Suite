# Architecture

## Modules

- `neo_pir_omr/core/engine.py`
  - Pipeline complet : chargement → détection document → warp → bbox grille → lecture OMR → protocole → scoring
  - Produit des **métriques de confiance** par item (`meta[item]["confidence"]`).

- `neo_pir_omr/ui/app_streamlit.py`
  - Interface Streamlit sécurisée : upload image + clé de scoring
  - Validation humaine assistée : revue/correction des items détectés comme *blank/ambiguous/low confidence*

- `neo_pir_omr/cli.py`
  - CLI batch pour traitement en masse et exports (CSV/JSON/PNG).

## Flux OMR (résumé)

1. **Détection du document** : contour externe 4 points + warp perspective.
2. **Auto‑orientation** : on teste 4 rotations et on choisit l’orientation la plus cohérente (grille + composantes haut‑gauche).
3. **Localisation du tableau 30×8** : projections horizontales/verticales sur masque de grille; fallback central robuste.
4. **Découpe cellules** : uniforme ou micro‑ajustée par pics de projection.
5. **Lecture marques** :
   - encre bleue (HSV) + encre noire (seuil grayscale) 
   - baseline adaptative pour réduire l’influence de l’impression
   - décision par top1/top2 + seuil marque + seuil ambiguïté.
6. **Protocole** : invalidation si trop de blancs / trop de "N"; imputation optionnelle.
7. **Scoring** : facettes + domaines à partir de la clé de scoring.
