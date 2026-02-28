# NEO PI‑R OMR Scanner (Professional Edition)

Un système professionnel, **sûr** et **précis** de scan de feuilles de réponses papier **remplies par des humains** (stylo bleu/noir), conçu pour le format NEO PI‑R (240 items, 5 options).

## Points forts

- **Détection robuste du document** (4 coins) + **correction de perspective**
- **Auto‑orientation** (0/90/180/270) par heuristiques de cohérence de grille
- **Localisation fiable du tableau 30×8** (évite QR/footer) + micro‑ajustement
- **Lecture OMR tolérante aux erreurs humaines** :
  - support **bleu + noir**
  - baseline adaptative pour l’encre noire (réduction “bruit” impression)
  - scores + **métriques de confiance** (gap top1/top2, blank, ambiguous)
- **Validation humaine assistée** : revue/correction des items *blank/ambiguous*
- **Traçabilité** : logs structurés, export CSV/JSON, rapport d’audit
- **Sécurité** : contrôles d’entrée (type/poids image), pas d’accès fichiers serveur via UI

---

## Installation

### 1) Créer un environnement

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Lancer l’interface Streamlit

```bash
streamlit run neo_pir_omr/ui/app_streamlit.py
```

### 3) Utiliser en ligne de commande (batch)

```bash
neo-pir-omr scan --image ./feuille.jpg --out ./resultats
neo-pir-omr batch --input ./photos --out ./resultats
```

---

## Sorties

Dans le dossier `--out` :
- `responses.csv` : réponses (item → option)
- `scores.json` : scores facettes/domaines + statut protocole
- `audit.json` : diagnostics (bbox, thresholds, items ambigus, etc.)
- `overlay.png` : visualisation (cellules, marques détectées)

---

## Fichiers de données

- `neo_pir_omr/data/scoring_key.csv` : clé de scoring (item, FD, D, N, A, FA)
- `neo_pir_omr/data/norms.csv` : normes (optionnel selon votre usage)

Vous pouvez fournir votre propre `scoring_key.csv` via l’UI ou la CLI.

---

## Sécurité (recommandations)

- Hébergez l’app derrière un reverse proxy (TLS) si usage multi‑utilisateur.
- Stockez les exports dans un répertoire dédié non‑public.
- Activez la rotation des logs et limitez la rétention.

---

## Licence

MIT
