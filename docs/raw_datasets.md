# Raw datasets

A factual inventory of everything currently under `data/raw/`, organized only by
whether the source ships with detection annotations (**labeled**) or not
(**unlabeled**). Counts are image files present on disk. This is a snapshot of the
raw material only — it makes no claim about which sources will be used, curated, or
how anything will be split.

Annotation formats present: Roboflow YOLO, plain per-scene YOLO, VIA polygon JSON,
COCO (JSON-lines), point-sighting CSV, and presence-only filenames.

## Labeled

### Aerial

| Dataset | Format | Images (on disk) | License | Source / reference |
|---|---|---|---|---|
| aerial-right-whale | Roboflow YOLO | 500 | CC BY 4.0 | https://universe.roboflow.com/whale-detect/aerial-right-whale-detection |
| whale-bbox-layer | Roboflow YOLO | 1,294 ¹ | CC BY 4.0 | https://universe.roboflow.com/whaleproject/whale-bbox-layer-ehkgx |
| dryad | VIA polygon JSON → bbox | 383 | CC0-1.0 | Gray et al. 2019 — https://datadryad.org/dataset/doi:10.5061/dryad.7482v2n |
| dryad_doi_10_5061_dryad_6q573n668__v20260312 (WHALESHAPE) | COCO (JSON-lines) | 468 ² | CC0-1.0 | Bagchi et al. 2025, DOI 10.5061/dryad.6q573n668 — https://datadryad.org/share/n6Z_l8PGx5Fji97H3opNqfZ8KHODa93G9-JmHojEc4A |
| uav-porpoises | per-scene YOLO txt (video frames) | 21,600 ³ | CC BY-NC 4.0 | Ptak et al. 2025 — https://zenodo.org/records/13267979 |

### Surface

| Dataset | Format | Images (on disk) | License | Source / reference |
|---|---|---|---|---|
| cetacean-detection | Roboflow YOLO | 3,632 ¹ | CC BY 4.0 | https://universe.roboflow.com/1-rfjla/cetacean-detection |
| cetacean-detector | Roboflow YOLO | 686 | CC BY 4.0 | https://universe.roboflow.com/whalecrop/cetacean-detector |
| dolphin | Roboflow YOLO | 1,005 | CC BY 4.0 | https://universe.roboflow.com/sysu-pc6gg/dolphin-i6izz |
| dolphin2 | Roboflow YOLO | 698 | CC BY 4.0 | https://universe.roboflow.com/nycuproject/dolphin-llvnv |
| whale-321z9 | Roboflow YOLO (generic class names) | 5,539 | CC BY 4.0 | https://universe.roboflow.com/eodudahs-gmail-com/whale-321z9 |
| whale-jybhw | Roboflow YOLO (single class `object-detection`) | 5,532 | CC BY 4.0 | https://universe.roboflow.com/chang-yup-son/whale-jybhw |

## Unlabeled

### Aerial

| Dataset | Format | Images (on disk) | License | Source / reference |
|---|---|---|---|---|
| gommapps-aerial-2017summer | images + sightings CSV (point sightings, not boxes) | 287 | Public domain (US Gov) | NOAA GoMMAPPS accession 0243469 (TO17 Summer) |
| gommapps-aerial-2018winter | images + sightings CSV (point sightings, not boxes) | 148 | Public domain (US Gov) | NOAA GoMMAPPS accession 0242273 (TO18 Winter) |
| gommapps-aerial-2018fall | images + sightings CSV (point sightings, not boxes) | 113 | Public domain (US Gov) | NOAA GoMMAPPS accession 0243468 (TO18 Fall) |
| MAMMALS | presence-only PNGs (taxon + certainty in filename, no boxes) | 1,398 | — | filenames encode `Cetacean`/`Dolphin`/`Seal` × `Definite`/`Probable`/`Possible` (e.g. `Harbour_porpoise`, `Common_dolphin`) |

## Footnotes

1. **Roboflow augmentation.** whale-bbox-layer and cetacean-detection export each
   source image as three baked-in augmented versions (original + flip + brightness):
   whale-bbox-layer 1,294 ≈ 431 originals; cetacean-detection 3,632 ≈ 1,211 originals.
2. **WHALESHAPE.** 468 images on disk (COCO `train.json` + `test.json`); the Dryad
   release documents 638 published whales (8,958 raw aerial frames in the study). COCO
   records carry bbox + segmentation + keypoints; `bbox_mode=1` (XYWH absolute px),
   `category_id=0`.
3. **uav-porpoises** is video-derived (drone tracking clips), hence many
   near-identical consecutive frames; a `tracking/` folder with per-scene track files
   sits alongside the frames.

**Commercial-use note:** uav-porpoises is non-commercial (CC BY-NC 4.0). The Roboflow
sources are CC BY 4.0; dryad / WHALESHAPE are CC0; gommapps is US-Gov public domain.
