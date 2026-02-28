# Security notes

## UI hardening

- L’UI n’autorise **pas** la lecture de fichiers arbitraires sur le serveur.
- Validation des fichiers uploadés (extension + taille).

## Operational security

- Déployer derrière HTTPS (reverse proxy) si accès distant.
- Ne pas exposer les exports dans un répertoire web public.
- Activer la rotation de logs (`neo_pir_omr/core/logging_conf.py`).
- En environnement multi‑utilisateur, isoler les sessions (conteneurs/VM) et appliquer des quotas.

## Data handling

Les feuilles de réponses peuvent être des données sensibles.
- Minimiser la conservation (rétention courte)
- Chiffrer au repos si nécessaire
- Contrôler les accès au stockage
