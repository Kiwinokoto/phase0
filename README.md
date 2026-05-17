# Artificial Life Sandbox — Phase 6

Prototype Python minimal pour générer et observer une planète 2D/3D avec proto-vie abstraite, interactions écologiques, mobilité et colonisation.

Phase 6 ne crée toujours **pas** de plantes, animaux, herbivores ou prédateurs codés en dur. Elle ajoute plutôt une mobilité populationnelle : les lignées peuvent se disperser passivement, migrer activement vers de meilleurs habitats, coloniser des fronts locaux, puis brancher plus facilement quand des colonies deviennent isolées.


## Écran d’accueil / setup planète

Le viewer démarre maintenant sur un écran de préparation avant de lancer la simulation.

À ce stade, la planète est déjà générée et visible à gauche, mais le temps ne tourne pas encore. Le panneau de droite affiche seulement les contrôles essentiels :

```text
- bouton Window / Fullscreen
- bouton View: 2D/3D
- seed utilisée
- statistiques résultantes de la planète générée
- paramètres modifiables de génération
- case `Skip formation intro` cochée par défaut
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

Par défaut, `Skip formation intro` est coché pour permettre les tests rapides : `Start simulation` lance directement la simulation. Décoche-le quand tu veux rejouer l'intro géologique déterministe, plus lente et purement visuelle. Elle raconte un fondu depuis le vide spatial, l'effondrement d'un nuage lumineux, une phase volcanique explosive, fumées/nuages primordiaux, pluies de condensation, puis révélation progressive des océans et continents. Cette intro ne modifie pas la planète générée, elle prépare simplement le départ de la simulation. Pendant l'intro, `Enter` / `Space` passe directement à la simulation.

Raccourcis utiles pendant le setup :

```text
Enter/Space   lancer la simulation ou skipper l’intro
r             seed aléatoire
f/F11         plein écran / fenêtre
g             bascule rendu carte 2D / planète 3D
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
bouton View: 2D/3D dans le panneau de droite
bouton Life: off/biomass/dominant dans le panneau de droite
bouton Weather: off/clouds/rain/all dans le panneau de droite
g      bascule rendu carte 2D / planète 3D
o      cycle aussi le life overlay
w      cycle aussi le weather overlay
r      nouvelle planète, seed aléatoire
s      capture d’écran
q/esc  quitter
```


## Rendu 2D / 3D

`View: 2D` conserve la carte équirectangulaire classique, pratique pour lire les couches.

`View: 3D` projette la même couche sur une planète en rotation lente. Ce n'est pas un moteur 3D physique : c'est un rendu orthographique logiciel de la texture 2D actuelle. Le clic d'inspection fonctionne aussi sur le disque visible de la planète ; les zones cachées au dos ne sont simplement pas cliquables.

En vue 3D, le fond spatial affiche maintenant un champ d'étoiles déterministe par seed. Il tourne plus lentement que la planète, avec un cycle complet sur la durée d'une année locale (`seasonal_period_ticks`).

## Couches affichables

```text
biome            vue synthétique jolie
elevation        relief
water            eau de surface / profondeur océanique
humidity         humidité écologique, moins redondante avec water
light            lumière dynamique selon latitude + saison
temperature      température dynamique latitude + altitude + saison + océan
clouds           voile nuageux procédural, dérive lente, couleur selon climat/volcanisme
rain             pluie stylisée + rares flashes d'orage sur zones humides/volcaniques
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
biotic_pressure  pression locale de consommation de biomasse vivante
migration_pressure  fronts récents de mobilité/colonisation
isolation_pressure  colonies de bord/îlots propices à la différenciation
```

## Overlays

Le viewer affiche un overlay de vie sur les couches non-vivantes, activé par défaut en mode `biomass`.

```text
Life: biomass   ajoute une lueur verte proportionnelle à la biomasse
Life: dominant  colore la vie selon la lignée dominante locale
Life: off       carte brute sans overlay
```

L'overlay ne remplace pas les couches `biomass`, `diversity` ou `dominant_life` : il sert seulement à voir la biosphère directement sur la carte jolie/abiotique.

Le bouton météo ajoute maintenant une atmosphère directement sur la couche `biome` :

```text
Weather: clouds  ajoute un voile nuageux plus patchy sur le biome
Weather: rain    ajoute nuages bas + pluie stylisée + rares flashes
Weather: all     combine nuages + pluie légère/storms
Weather: off     carte biome brute sans atmosphère
```

La météo reste visuelle : elle est conséquence des champs existants, pas une cause écologique. Elle varie avec le tick et la saison pour éviter une pluie figée en permanence. Les années sont maintenant de durée seed-dérivée par défaut, donc toutes les planètes ne tournent pas sur le même calendrier de 2400 ticks.

## Logique Phase 6

La vie reste abstraite et populationnelle : une lignée n'est pas un individu, c'est un champ de population local.

Chaque lignée a des traits héritables :

```text
photosynthesis        capte lumière + nutriments
chemosynthesis        exploite l'énergie chimique
organic_absorption    exploite la matière morte
living_consumption    extrait de l'énergie de biomasse vivante vulnérable
defense               réduit la pression biotique subie
storage               amortit famine/stress mais coûte à maintenir
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
8. les populations se dispersent passivement ;
9. les populations mobiles migrent légèrement vers les cellules voisines plus favorables ;
10. les fronts de colonisation alimentent `migration_pressure` ;
11. les colonies de bord/îlots alimentent `isolation_pressure` ;
12. des branches mutantes apparaissent parfois, avec un bonus si la lignée a des colonies isolées ;
13. une pression biotique locale apparaît quand des lignées exploitent d'autres biomasses ;
14. défense et stockage réduisent une partie de cette pression ;
15. les lignées trop faibles s'éteignent.
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
biotic_pressure  zones où la vie exerce une pression sur la vie
migration_pressure  fronts de déplacement/colonisation récents
isolation_pressure  colonies marginales pouvant brancher en descendants
clouds/rain     couches atmosphériques dédiées + overlay météo sur biome
nutrients       ressources consommées puis recyclées
```

La couche `dead_matter` peut rester sombre au tout début. En Phase 5 elle devient plus visible après les premiers crashs, extinctions locales ou zones de surpopulation. L'affichage est auto-scalé : une petite quantité de débris peut devenir visible même si la valeur brute reste faible.


## Atmosphere visual layers

Deux couches visuelles ont été ajoutées à la carte de simulation, hors intro :

```text
clouds  voile nuageux procédural ; opacité variable ; dérive lente avec le tick
rain    pluie stylisée ; intensité liée à humidité/eau/température ; flashes rares près des zones humides/volcaniques
```

Ces couches sont volontairement **visuelles seulement** pour l'instant. Elles ne modifient pas encore les ressources, l'humidité ou la fertilité. Elles servent à rendre l'atmosphère observable avant d'en faire éventuellement un vrai système météo plus tard. Le life overlay est désactivé sur ces deux couches pour garder la lecture atmosphérique propre.

La couche `biome` peut maintenant recevoir un overlay `Weather: clouds` ou `Weather: rain`. Les précipitations dérivent avec le temps, les saisons et des cycles de tempête procéduraux ; elles ne restent donc pas strictement identiques d'un tick à l'autre. Le panneau affiche aussi `year/day`, une indication de saison et la moyenne cloud/rain pour aider à comprendre le cycle en cours.

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
- traits Phase 5 `living_consumption`, `defense`, `storage` ;
- couches Phase 6 `migration_pressure` et `isolation_pressure` ;
- couche `biotic_pressure` visible lorsque la vie interagit avec la vie ;
- évolution déterministe à seed et steps identiques ;
- rendu d'intro géologique déterministe et sans effet sur la simulation ;
- overlays météo sur biome, dérive temporelle et labels de saison pour l'observateur.

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

## Phase 5 genealogy modal

When a lineage card is open, the `Open genealogy tree` button opens a modal view for that lineage.

It shows:

```text
- ancestral line from root lineage to the selected lineage
- direct children count
- total descendant count
- descendant rows with indentation by generation
- current biomass and creation tick for each visible row
```

Rows inside the modal are clickable: selecting another lineage recenters the modal on that lineage and updates the map highlight. `Esc`, the close button, or clicking outside the modal closes it. This is observational UI only; it does not affect simulation dynamics.


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

The setup screen now defaults back to fast iteration: `Skip formation intro` starts checked. Uncheck it when you want the slower contemplative geological prelude, where the cloud/accretion, primordial fire, rain/condensation and young-planet stages have time to read visually.

The right panel section headers are more separated and visually anchored, and the runtime layer legend can now be collapsed like the other sections.


## Phase 5 observer polish / intro v2

Small observer polish before continuing ecology work:

```text
- Simulation, Planet averages and Life summary sections start collapsed by default
- Event log has an `Open event summary` button and modal
- event summary modal shows counts plus recent major world events
- event summary modal has filters: volcanoes hidden/shown and births early-only/hidden/all
- branches/speciation and extinctions remain visible even when filters hide later abiogenesis noise
- selected extinct lineages remain selected instead of visually disappearing
- extinct lineages are shown in crimson in cards, rows and genealogy trees
- the selected map label stays visible even when the selected lineage has no remaining population
- geological intro v2 adds black fade, stellar cloud collapse, explosive volcanism, smoke/cloud layer, rain and slower ocean/continent reveal
```

This patch remains visual/observer-only: it does not change simulation balance or generated planet fields.

## Phase 5 adjustment — clearer lineage tracking

This polish pass makes descendant/speciation events easier to catch while keeping the map readable:

```text
- layer legend starts folded by default
- biomass overlay now uses distinct tones for ocean life and terrestrial life
- Planet averages includes rough bio land / bio ocean share
- event summary can still show all root births and all extinctions
- branch/speciation events are highlighted in yellow and prefixed as descendants
- event history keeps more entries so older births/extinctions are less likely to disappear
- abiogenesis now reserves part of the lineage capacity for later descendants
- speciation is slightly less rare, so genealogy should become visible earlier
```

Root births are still independent abiogenesis events. True descendants appear as `branch` events and have a `parent_id` in the species card/genealogy tree.


## Phase 5 3D star field

- fond d'étoiles visible uniquement en rendu `View: 3D` ;
- étoiles générées de manière déterministe depuis la seed ;
- rotation lente du ciel sur un cycle d'une année locale ;
- purement visuel : aucun effet sur saisons, météo ou simulation.


## Phase 6 mobility / colonization

Phase 6 ajoute une mobilité encore abstraite, mais plus directionnelle que la simple diffusion :

```text
- dispersal reste un trait héritable ;
- la diffusion passive continue d'étaler les populations ;
- une migration active déplace une petite fraction vers des cellules voisines plus adaptées ;
- chaque déplacement coûte un peu de biomasse/énergie ;
- migration_pressure montre les fronts récents de mouvement/colonisation ;
- isolation_pressure met en évidence les colonies de bord ou d'îlot ;
- la spéciation est légèrement biaisée vers ces colonies isolées.
```

Ce n'est toujours pas un modèle d'individus. Une lignée reste un champ de population. L'objectif est de rendre la géographie plus importante : îles, côtes, fronts d'expansion et niches séparées doivent commencer à produire des branches généalogiques plus lisibles.
