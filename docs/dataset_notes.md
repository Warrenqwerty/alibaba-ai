## DeepFashion2 For 3.1.1

DeepFashion2 is the first dataset target for clothing detection and instance
segmentation.

Local repo layout:

- Root: `data/DeepFashion2`
- Train images: `data/DeepFashion2/train/image`
- Train annotations: `data/DeepFashion2/train/annos`
- Validation images: `data/DeepFashion2/validation/image`
- Validation annotations: `data/DeepFashion2/validation/annos`

AutoDL layout:

- Train images: `/root/autodl-tmp/datasets/DeepFashion2/train/image`
- Train annotations: `/root/autodl-tmp/datasets/DeepFashion2/train/annos`
- Validation images: `/root/autodl-tmp/datasets/DeepFashion2/validation/image`
- Validation annotations: `/root/autodl-tmp/datasets/DeepFashion2/validation/annos`

The adapter maps DeepFashion2 categories into the PRD taxonomy:

- top: short/long sleeve tops, vests, sling tops
- pants: shorts, trousers
- skirt: skirts
- outerwear: short/long sleeve outerwear
- dress: short/long sleeve dresses, vest dresses, sling dresses

Shoes, bags, and accessories are reserved in the model head for PRD completeness;
they require additional labeled data beyond DeepFashion2.

See `docs/get_3_1_1_data.md` for dataset access status and AutoDL commands.
