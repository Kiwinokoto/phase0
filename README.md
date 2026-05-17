# Artificial Life Sandbox — Phase 2

Prototype Python minimal pour générer et observer une planète 2D.

Phase 2 contient encore **zéro vie**. Le but est de rendre le substrat dynamique avant d'ajouter des réplicateurs : saisons simples, diffusion, recharge/dissipation des ressources, toxicité qui évolue, et impulsions volcaniques.

Ce que cette phase ajoute par rapport à la Phase 1 :

- saisons simples : lumière/température changent avec le temps ;
- nutriments dynamiques : diffusion, recharge par sources, lessivage ;
- énergie chimique dynamique : recharge autour du volcanisme, diffusion, dissipation ;
- toxicité dynamique : production autour des zones actives, diffusion, dissipation ;
- impulsions volcaniques temporaires ;
- fertilité recalculée à chaque tick ;
- viewer vraiment animé : la vitesse de tick a maintenant un effet visible ;
- bascule fenêtre/plein écran ;
- légende contextuelle pour chaque couche.

## Installation / lancement rapide

```bash
cd alife_phase2
./run.sh
```

`run.sh` active/crée automatiquement l’environnement Conda, installe les dépendances seulement si `requirements.txt` a changé, puis lance `python run.py`.

Le script crée l'environnement Conda `alife_phase2` s'il n'existe pas. Ensuite il garde un hash local de `requirements.txt` dans `.alife_env/` pour éviter de refaire `pip install` à chaque lancement.

Options utiles :

```bash
source ./create_or_activate_env.sh --no-install
source ./create_or_activate_env.sh --force-install
source ./create_or_activate_env.sh --name alife-dev
source ./create_or_activate_env.sh --python 3.11
```

Important : `create_or_activate_env.sh` doit être lancé avec `source`, pas `./create_or_activate_env.sh`, sinon l'activation Conda ne reste pas dans le terminal courant.

## Lancer

```bash
./run.sh
```

Options utiles :

```bash
./run.sh --seed 123 --width 320 --height 160 --scale 4 --sea-level 0.52
python run.py --seed 123 --width 320 --height 160 --scale 4 --sea-level 0.52
```

Par défaut, `python run.py` et `./run.sh` génèrent une seed aléatoire.
Utilise `--seed 123` seulement quand tu veux reproduire exactement la même planète.

## Contrôles

```text
space  pause/reprise
tab    couche suivante
←/→    couche précédente/suivante
↑/↓    vitesse de tick
+/-    vitesse de tick aussi
bouton Fullscreen/Window dans le panneau de droite
r      nouvelle planète, seed aléatoire
s      capture d’écran
q/esc  quitter
```

## Couches affichables

```text
biome            vue synthétique jolie
elevation        relief
water            eau de surface / profondeur océanique
humidity         humidité écologique, moins redondante avec water
light            lumière dynamique selon latitude + saison
temperature      température dynamique latitude + altitude + saison + océan
volcanism        volcanisme de base + impulsions temporaires
minerals         ressources minérales statiques
nutrients        nutriments dynamiques, surtout côtes/basses terres/érosion
chemical_energy  énergie chimique dynamique, surtout zones volcaniques
toxicity         hostilité chimique/thermique dynamique
fertility        potentiel global pour une future proto-vie, recalculé en continu
```

Le panneau de droite affiche maintenant une légende contextuelle pour la couche active : soit une barre de gradient avec labels, soit des couleurs catégorielles pour `biome`.

## Logique Phase 2

La planète n'a toujours pas de vie, mais elle bouge :

```text
saisons                → lumière/température varient selon hémisphère
volcanisme actif       → énergie chimique + toxicité temporaires
nutriments             → diffusion + recharge + lessivage lent
énergie chimique       → recharge + diffusion + dissipation
toxicité               → production + diffusion + dissipation
fertilité              → nutriments + eau/humidité + énergie + température - toxicité
```

L'objectif visuel est de pouvoir repérer des niches instables :

```text
zones fertiles mais temporaires
sources chimiques actives
zones toxiques qui se calment
côtes stables
intérieurs pauvres
hémisphères qui changent avec la saison
```

## Tests

```bash
pytest
```

ou :

```bash
python -m pytest
```

Les tests vérifient notamment :

- dimensions des couches ;
- champs normalisés entre 0 et 1 ;
- déterminisme avec une seed fixe ;
- équateur plus lumineux/chaud que les pôles au tick initial ;
- énergie chimique corrélée au volcanisme ;
- toxicité pénalisant la fertilité ;
- humidité pas simplement copiée sur `water` ;
- `step()` modifie les champs dynamiques ;
- les champs statiques restent stables ;
- deux planètes identiques évoluent pareil avec la même seed.

## Prochaines phases prévues

Phase 3 : premiers réplicateurs simples.

- apparition rare selon fertilité/stabilité ;
- énergie interne ;
- reproduction ;
- mort ;
- mutation légère ;
- premières lignées.

Phase 4 : évolution réellement exploitable.

- traits héritables ;
- divergence locale ;
- spéciation ;
- extinctions ;
- arbre des lignées.

## Notes Linux / Mesa / GLX

Si tu vois une erreur du type `MESA-LOADER`, `iris_dri.so`, `swrast_dri.so` ou `GLXCreateContext`, lance plutôt :

```bash
./run.sh
```

Le lanceur force par défaut un rendu logiciel SDL/Pygame pour éviter les chemins OpenGL fragiles sur certains laptops/VM Linux.

Si l’erreur persiste, il manque probablement des drivers Mesa côté système :

```bash
sudo apt update
sudo apt install -y libgl1-mesa-dri libglx-mesa0 mesa-utils
```

Pour réessayer le rendu matériel :

```bash
ALIFE_FORCE_SOFTWARE_RENDER=0 ./run.sh
```
