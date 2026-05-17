# Artificial Life Sandbox — Phase 3

Prototype Python minimal pour générer et observer une planète 2D avec premières proto-formes de vie abstraites.

Phase 3 ajoute la première dynamique vraiment évolutive : des lignées apparaissent rarement dans des niches fertiles, se développent en champs de population, meurent, recyclent de la matière, s'éteignent ou produisent parfois des branches mutantes.

Important : il n'y a toujours **pas** de plantes, animaux, herbivores ou prédateurs codés en dur. On garde seulement des capacités abstraites : capter la lumière, exploiter l'énergie chimique, absorber de la matière morte, tolérer température/eau/toxicité, se disperser et muter.

## Installation / lancement rapide

```bash
./run.sh
```

`run.sh` active/crée automatiquement l’environnement Conda, installe les dépendances seulement si `requirements.txt` a changé, puis lance `python run.py`.

Le script crée l'environnement Conda `alife_phase3` s'il n'existe pas. Ensuite il garde un hash local de `requirements.txt` dans `.alife_env/` pour éviter de refaire `pip install` à chaque lancement.

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
fertility        potentiel global pour abiogenesis/proto-vie
dead_matter      matière organique morte recyclée en nutriments
biomass          densité totale de proto-vie vivante
diversity        coexistence locale de plusieurs lignées
dominant_life    couleur de la lignée dominante + densité de biomasse
```

## Logique Phase 3

La vie reste abstraite et populationnelle : une lignée n'est pas un individu, c'est un champ de population local.

Chaque lignée a des traits héritables :

```text
photosynthesis        capte lumière + nutriments
chemosynthesis        exploite l'énergie chimique
organic_absorption    exploite la matière morte
temperature_optimum   température préférée
temperature_tolerance largeur de tolérance thermique
water_preference      besoin en eau/humidité
toxicity_tolerance    résistance à la toxicité
reproduction_rate     vitesse de croissance
metabolism_cost       coût de maintien
dispersal             diffusion spatiale
mutation_rate         probabilité de branchement évolutif
```

À chaque tick :

```text
1. climat/saisons/ressources se mettent à jour ;
2. abiogenesis rare dans les niches fertiles ;
3. chaque lignée gagne/perd de la population selon énergie, eau, température, toxicité ;
4. les morts alimentent dead_matter ;
5. dead_matter recycle lentement des nutriments ;
6. les populations se dispersent légèrement ;
7. des branches mutantes apparaissent parfois ;
8. les lignées trop faibles s'éteignent.
```

## Ce qu'il faut observer

Les couches les plus importantes :

```text
fertility       où la vie a le plus de chances d'apparaître
biomass         où elle réussit actuellement
dominant_life   quelle lignée domine localement
diversity       où plusieurs lignées coexistent
dead_matter     traces d'effondrement/recyclage
```

Le panneau de droite affiche aussi les lignées principales, leur biomasse et une interprétation fonctionnelle approximative. Cette interprétation n'est pas une classe dans le code : elle est déduite des traits.

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
- dynamique abiotiques toujours stable ;
- apparition de proto-lignées dans un monde fertile ;
- champs de vie bornés ;
- évolution déterministe à seed et steps identiques.

## Notes Linux / Mesa / GLX

Si tu vois une erreur du type `MESA-LOADER`, `iris_dri.so`, `swrast_dri.so` ou `GLXCreateContext`, lance plutôt :

```bash
./run.sh
```

Le lanceur force par défaut un rendu logiciel SDL/Pygame pour éviter les chemins OpenGL fragiles sur certains laptops/VM Linux.

Pour réessayer le rendu matériel :

```bash
ALIFE_FORCE_SOFTWARE_RENDER=0 ./run.sh
```
