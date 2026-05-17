# Artificial Life Sandbox — Phase 4

Prototype Python minimal pour générer et observer une planète 2D avec proto-vie abstraite et contraintes écologiques.

Phase 4 ne crée toujours **pas** de plantes, animaux, herbivores ou prédateurs codés en dur. Elle renforce plutôt le moteur d'évolution : la vie consomme des ressources pour croître et se maintenir, produit de la matière morte, subit la surpopulation locale, peut stagner, s'effondrer ou s'éteindre.


## Écran d’accueil / setup planète

Le viewer démarre maintenant sur un écran de préparation avant de lancer la simulation.

À ce stade, la planète est déjà générée et visible à gauche, mais le temps ne tourne pas encore. Le panneau de droite affiche seulement les contrôles essentiels :

```text
- bouton Window / Fullscreen
- seed utilisée
- statistiques résultantes de la planète générée
- paramètres modifiables de génération
- case `Skip formation intro` décochée par défaut
- bouton Start simulation en bas
```

Les boutons `+` / `-` modifient immédiatement la planète prévisualisée. Chaque paramètre a aussi une barre colorée : clique ou glisse directement sur la barre pour régler rapidement la valeur entre son minimum et son maximum. Le bouton `Random` choisit une nouvelle seed.

Paramètres modifiables dans le setup :

```text
sea level
continent scale
volcanism
equator temperature
pole temperature

Terrain detail — deterministic
detail octaves
detail gain
```

Les deux paramètres de détail restent regroupés en bas : ils changent la texture/fractale du relief, mais ne sont pas randomisés automatiquement.

Quand la planète te plaît, clique `Start simulation` ou appuie sur `Enter` / `Space`.

Par défaut, `Skip formation intro` est décoché : `Start simulation` joue une intro géologique déterministe, plus lente et purement visuelle. Elle raconte la transition accrétion / magma ocean / pluies intenses / refroidissement / planète jeune stable. Cette intro ne modifie pas la planète générée, elle prépare simplement le départ de la simulation. Coche `Skip formation intro` quand tu veux tester vite. Pendant l'intro, `Enter` / `Space` passe directement à la simulation.

Raccourcis utiles pendant le setup :

```text
Enter/Space   lancer la simulation ou skipper l’intro
r             seed aléatoire
f/F11         plein écran / fenêtre
s             screenshot
q/esc         quitter
```

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

## Contrôles après Start

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
- évolution déterministe à seed et steps identiques ;
- rendu d'intro géologique déterministe et sans effet sur la simulation.

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

The right panel is now grouped from general to particular instead of being one long list:

```text
1. Active layer
2. Main buttons and compact controls
3. Simulation status
4. Planet averages
5. Life summary and global top lineages
6. Selected zone inspector
7. Selected lineage / habitat card
8. Current layer legend, pinned near the bottom
```

Runtime sections are collapsible: click a section header to fold or unfold it. This keeps the panel usable as more ecology and lineage data is added. The legend stays available near the bottom because it explains the currently visible layer.

The panel also includes a compact event log for major observable events: abiogenesis, lineage branching, extinctions, and volcanic pulses.

## Phase 4 species cards patch

The local and global lineage rows are now clickable.

Recommended observation flow:

```text
1. click a living zone on the map
2. inspect the local top lineages in the right panel
3. click one lineage row
4. read the selected lineage card
5. follow its highlighted distribution on the map
```

The selected lineage card shows:

```text
- name, strategy label and color
- living/extinct status
- age, parent and descendant count
- current biomass and historical peak
- occupied cells and strongest cell
- habitat summary: temperature, water access, fertility, toxicity, nutrients, chemical energy, dead matter and light
- core traits: photosynthesis, chemosynthesis, detritus absorption, dispersal, toxicity tolerance, mutation rate, temperature optimum/tolerance
```

This is still observational UI only: selecting or highlighting a lineage does not affect the simulation.


## Phase 4 setup screen patch

The app now opens on a planet setup screen instead of immediately starting the simulation. The generated planet is visible as before, but the right panel is simplified until `Start simulation` is pressed.

This keeps the later ecology UI uncluttered while allowing quick world iteration: seed, sea level, continent scale, terrain detail, volcanism and temperature bounds regenerate the preview immediately.

## Phase 4 setup sliders / collapsible panel patch

The setup screen now exposes planet parameters as direct manipulation sliders. Click or drag a colored bar to regenerate the preview immediately. The old `+` / `-` buttons are still present for precise nudges.

After `Start simulation`, the right panel is ordered from general to particular:

```text
Simulation → Planet averages → Life summary → Selected zone → Selected lineage / habitat
```

Click any section header to collapse or expand it. The layer legend remains pinned near the bottom of the panel so it does not get buried behind lineage details; it is collapsible too when you need extra room.


## Phase 4 finish polish

Small final Phase 4 polish before moving to richer ecology:

```text
- setup sliders now keep deterministic terrain-detail controls grouped at the bottom
- runtime panel adds a compact collapsible event log
- event log records births, branches, extinctions and volcanic pulses
- recent events are exposed via Planet.recent_events(limit=...) for tests/UI
```


## Phase 4 geological intro UX polish

The setup screen now treats the geological prelude as the default experience: `Skip formation intro` starts unchecked. The intro is slower and more contemplative, so the cloud/accretion, primordial fire, rain/condensation and young-planet stages have time to read visually.

The right panel section headers are more separated and visually anchored, and the runtime layer legend can now be collapsed like the other sections.
