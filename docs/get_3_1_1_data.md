## Data For 3.1.1 Fashion Instance Segmentation

### Primary Dataset: DeepFashion2

DeepFashion2 is the best match for the current training adapter because it
contains clothing boxes and instance masks. It is not fully public-direct:

- Official repo: `https://github.com/switchablenorms/DeepFashion2`
- Official Google Drive folder: `https://drive.google.com/drive/folders/125F48fsMBz2EF0Cpqk6aaHet5VH399Ok?usp=sharing`
- Access note: the official repo asks users to complete a Google Form to obtain
  the dataset password. I cannot automatically fetch the archive without that
  password/account access.

After you obtain the files, put them on AutoDL like this:

```text
/root/autodl-tmp/datasets/DeepFashion2/
  train/
    image/
    annos/
  validation/
    image/
    annos/
```

Then verify the expected paths:

```bash
ls /root/autodl-tmp/datasets/DeepFashion2/train/image | head
ls /root/autodl-tmp/datasets/DeepFashion2/train/annos | head
ls /root/autodl-tmp/datasets/DeepFashion2/validation/image | head
ls /root/autodl-tmp/datasets/DeepFashion2/validation/annos | head
```

### Public Fallback Dataset: Fashionpedia

Fashionpedia is publicly downloadable from S3 and includes instance segmentation
annotations for richer fashion categories, including categories that can support
the PRD classes for shoes, bags, and accessories.

On AutoDL:

```bash
cd /root/alibaba-ai
python scripts/data/download_fashionpedia.py \
  --root /root/autodl-tmp/datasets/Fashionpedia
```

The public annotation URL was reachable from this environment when network was
allowed. The full image archives are large, so download them on AutoDL instead
of the Mac.

### Current Recommendation

Use DeepFashion2 first if you can access it, because the current training code
already supports its annotation format. Keep Fashionpedia as the public fallback
and as the source to extend coverage for shoes, bags, and accessories.
