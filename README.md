# Artificial Life Sandbox — Phase 4

Prototype Python minimal pour générer et observer une planète 2D avec proto-vie abstraite et contraintes écologiques.

Phase 4 ne crée toujours **pas** de plantes, animaux, herbivores ou prédateurs codés en dur. Elle renforce plutôt le moteur d'évolution : la vie consomme des ressources pour croître et se maintenir, produit de la matière morte, subit la surpopulation locale, peut stagner, s'effondrer ou s'éteindre.

## Installation / lancement rapide

```bash
./run.sh
```

`run.sh` active/crée automatiquement l’environnement Conda, installe les dépendances seulement si `requirements.txt` a changé, puis lance `python run.py`.

Le script garde par défaut l'environnement Conda `alife_phase3` pour éviter de recréer un env à chaque phase. Tu peux changer avec :

```bash
ALIFE_CONDA_ENV=alife ./run.sh
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
bouton Life: off/biomass/dominant dans le panneau de droite
o      cycle aussi le life overlay
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
dead_matter      matière morte/recyclée ; affichage auto-scalé pour être lisible
biomass          densité totale de proto-vie vivante
diversity        coexistence locale de plusieurs lignées
dominant_life    couleur de la lignée dominante + densité de biomasse
```

## Life overlay

Le viewer affiche un overlay de vie sur les couches non-vivantes, activé par défaut en mode `biomass`.

```text
Life: biomass   ajoute une lueur verte proportionnelle à la biomasse
Life: dominant  colore la vie selon la lignée dominante locale
Life: off       carte brute sans overlay
```

L'overlay ne remplace pas les couches `biomass`, `diversity` ou `dominant_life` : il sert seulement à voir la biosphère directement sur la carte jolie/abiotique.

## Logique Phase 4

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
4. la biomasse locale est limitée par une capacité de charge écologique ;
5. les populations paient un coût de maintenance en ressources ;
6. la surpopulation, la famine et le stress produisent de la matière morte ;
7. dead_matter recycle lentement des nutriments ;
8. les populations se dispersent légèrement ;
9. des branches mutantes apparaissent parfois ;
10. les lignées trop faibles s'éteignent.
```

## Ce qu'il faut observer

Les couches les plus importantes :

```text
fertility       où la vie a le plus de chances d'apparaître
biomass         où elle réussit actuellement
biome + overlay voir la vie sur la carte principale
dominant_life   quelle lignée domine localement
diversity       où plusieurs lignées coexistent
dead_matter     traces d'effondrement/recyclage
nutrients       ressources consommées puis recyclées
```

La couche `dead_matter` peut rester sombre au tout début. En Phase 4 elle devient plus visible après les premiers crashs, extinctions locales ou zones de surpopulation. L'affichage est auto-scalé : une petite quantité de débris peut devenir visible même si la valeur brute reste faible.

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
- dynamique abiotique stable ;
- apparition de proto-lignées dans un monde fertile ;
- champs de vie bornés ;
- production de matière morte après croissance/stress ;
- population contrainte par les ressources ;
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


## Phase 4 inspector patch

The viewer now starts fullscreen by default. Use `--windowed` for the old startup mode:

```bash
./run.sh --windowed
```

Click anywhere on the map to select a local zone. The right panel shows local biomass, diversity, fertility/toxicity, dead matter, and the dominant lineages in that zone. The active layer name is also shown prominently under the phase title.

## Phase 4 right-panel readability patch

The right panel is now grouped by priority instead of being one long list:

```text
1. Active layer
2. Main buttons and compact controls
3. Simulation status
4. Selected zone inspector
5. Life summary and global top lineages
6. Planet averages
7. Current layer legend
```

The selected zone inspector is intentionally placed high in the panel because it will remain useful in later phases when there are more lineages, strategies and signals to track.
