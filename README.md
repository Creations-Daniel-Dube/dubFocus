# dubFocus

> Application du catalogue dubSuite. Installation, mises à jour et désinstallation : [dubSuite](https://github.com/Creations-Daniel-Dube/dubSuite).

dubFocus raffine la mise au point d’images astronomiques. Il estime le flou — disque de défocalisation ou gaussienne — puis le retire sur un GPU NVIDIA, pour redonner du contraste aux étoiles et aux détails fins.

Le calcul porte sur la luminance. La couleur d’origine est réappliquée ensuite, ce qui évite les franges colorées qu’une déconvolution canal par canal produit souvent sur le ciel.

## Ce que fait le traitement

1. Un filtre de Wiener fournit un premier redressement, borné pour limiter le bruit amplifié.
2. Richardson-Lucy itère à partir de cette base. Chaque passe compare l’image floue observée à une reconvolution de l’estimation courante et corrige l’écart.
3. Une régularisation laplacienne et un amortissement freinent le bruit dans les zones faibles.
4. Un anti-ringing atténue les anneaux qui apparaissent autour des étoiles brillantes et des bords contrastés.

Pendant le déplacement d’un curseur, les grandes images sont traitées en aperçu réduit (grand côté ramené vers 2048 pixels). Le bouton **Raffiner** relance Richardson-Lucy à pleine résolution.

## Utilisation

Ouvrez une image, réglez la PSF jusqu’à ce que le flou disparaisse sans faire naître d’anneaux, puis enregistrez.

| Réglage | Rôle |
| --- | --- |
| Disque de défocalisation | Flou en disque, typique d’une mise au point décalée. Le rayon est en pixels. |
| Gaussienne | Flou plus doux. Le sigma est en pixels. |
| Itérations Richardson-Lucy | De 1 à 80. Plus d’itérations affine le détail et peut aussi faire monter le bruit. |
| Régularisation K | De 10⁻⁴ à 10⁻¹. Une valeur plus haute lisse davantage. |
| Amortissement | Réduit les corrections là où le signal est faible. |
| Anti-ringing | Mélange le résultat avec une version plus douce près des forts contrastes (0 à 100 %). |
| Interpréter comme sRGB | Décode le gamma d’affichage avant le calcul. Proposé automatiquement pour les fichiers 8 bits. |
| Noir, blanc, gamma | Niveaux d’affichage et d’export, appliqués à l’original et au résultat. |
| Balayage | À gauche l’original, à droite le résultat. La barre orange se déplace aussi à la souris. |

La vue se déplace au glisser, le zoom suit la molette, et un double-clic recadre l’image.

Les réglages sont conservés dans `%APPDATA%\dubFocus\settings.json` sous Windows, ou dans `$XDG_CONFIG_HOME/dubFocus/settings.json` ailleurs.

## Formats

**Lecture :** JPEG, PNG et TIFF, en niveaux de gris ou RVB, 8 ou 16 bits. Le canal alpha est ignoré.

**Enregistrement :**

- PNG 8 bits, PNG 16 bits, JPEG et TIFF 16 bits suivent les niveaux affichés à l’écran.
- TIFF 32 bits reste en valeurs linéaires, niveaux compris.

## Prérequis

Un GPU NVIDIA et un pilote exposant CUDA 12, dont cuFFT. Sans GPU CUDA, l’application s’ouvre et signale que le traitement est indisponible.

La compilation des noyaux GPU demande Rust nightly avec la cible `nvptx64-nvidia-cuda` :

```text
rustup toolchain install nightly
rustup +nightly target add nvptx64-nvidia-cuda
cargo build --release
```

L’exécutable produit est `target/release/dubFocus.exe`.
