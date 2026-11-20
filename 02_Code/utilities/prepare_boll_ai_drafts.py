"""Conservative, unverified machine drafts for HUMAN review; never score these.

This does not import or reproduce the evaluated B rules. It uses a separate
brightness/chromaticity region-growing heuristic, with manually identified
mosaic-artifact regions withheld. Similar colour cues can still cause correlated
errors: these drafts are NOT independent ground truth, even though the rules differ.
"""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / '00_Project/annotation'

# Coordinates inspected in the original, unscaled 384-pixel tiles (x0,y0,x1,y1).
# Whole ambiguous rectangles are withheld, including any real fiber within them.
ARTIFACTS = {
    # Bright soil corridor: some nearly neutral soil is confused with fiber.
    '2023_F1_N000_P07': [(95,30,205,135),(110,165,195,255),(100,275,215,384)],
    '2024_F1_N000_P07': [(0,145,87,225),(45,226,180,365)],
    '2024_F1_N150_P19': [(25,0,245,155),(175,145,280,210)],
    '2024_F1_N225_P05': [(0,0,75,175)],
    '2024_F2_N000_P07': [(0,65,375,185),(0,210,175,350),(235,200,365,384)],
    '2024_F2_N075_P14': [(0,0,120,65),(0,190,160,240)],
    '2024_F2_N150_P19': [(0,0,90,384),(90,0,215,85)],
    '2024_F2_N300_P10': [(0,0,100,210),(165,145,250,275)],
    '2024_F2_N450_P06': [(0,0,175,275)],
}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    masks = DEST / 'ai_draft_masks'
    previews = DEST / 'ai_draft_previews'
    masks.mkdir(exist_ok=True)
    previews.mkdir(exist_ok=True)
    rows = []
    with (DEST/'sampling_manifest.csv').open() as f:
        manifest = list(csv.DictReader(f))
    for r in manifest:
        name = r['image_id']
        original = DEST/'images'/f'{name}.png'
        assert sha(original) == r['rgb_sha256']
        rgb = np.asarray(Image.open(original).convert('RGB'))
        valid = np.asarray(Image.open(DEST/'valid_masks'/f'{name}.png')) > 0
        x = rgb.astype(float)
        low, high = x.min(axis=2), x.max(axis=2)
        brightness = x.mean(axis=2)
        chroma = high-low
        saturation = chroma/np.maximum(high,1)
        seed = (low >= 175) & (chroma <= 52) & valid
        grow = (brightness >= 142) & (low >= 110) & (chroma <= 53) & (saturation <= .30) & valid
        regions = ndi.binary_propagation(seed, mask=grow)
        labels, _ = ndi.label(regions)
        size = np.bincount(labels.ravel())
        regions &= size[labels] >= 12
        # One-pixel ambiguous mixed edge around every proposed white region.
        core = ndi.binary_erosion(regions, border_value=0)
        uncertain_edge = ndi.binary_dilation(regions, iterations=1) & ~core
        plausible_dim_fiber = (brightness >= 100) & (saturation <= .35)
        lab = np.zeros(valid.shape, dtype=np.uint8)
        lab[plausible_dim_fiber | uncertain_edge] = 255
        lab[core] = 1
        withheld = np.zeros(valid.shape, dtype=bool)
        for x0,y0,x1,y1 in ARTIFACTS.get(name,[]):
            withheld[y0:y1,x0:x1] = True
        lab[withheld | ~valid] = 255
        mask_path = masks/f'{name}.png'
        Image.fromarray(lab).save(mask_path)
        overlay = rgb.astype(float).copy()
        for value, color, alpha in [(1, (0,180,255), .55),(255,(240,160,30),.30)]:
            select = (lab == value) & valid
            overlay[select] = (1-alpha)*overlay[select] + alpha*np.array(color)
        overlay[~valid] = 0
        preview = Image.new('RGB',(768,420),'white')
        preview.paste(Image.fromarray(rgb),(0,36))
        preview.paste(Image.fromarray(overlay.astype('uint8')),(384,36))
        draw = ImageDraw.Draw(preview)
        draw.text((8,4),name+' | UNREVIEWED MACHINE DRAFT',fill='black')
        draw.text((8,20),'Original RGB',fill='black')
        draw.text((392,20),'Cyan=fiber draft; amber=unknown; no tint=background',fill='black')
        preview.save(previews/f'{name}.png')
        counts = {str(k):int(((lab==k)&valid).sum()) for k in [0,1,255]}
        rows.append(dict(image_id=name,domain=r['domain'],N=int(r['N']),
                         status='AI_DRAFT_AWAITING_FULL_HUMAN_REVIEW',
                         source_sha256=sha(original),draft_sha256=sha(mask_path),
                         valid_pixels=int(valid.sum()),background_pixels=counts['0'],
                         proposed_fiber_pixels=counts['1'],unknown_pixels=counts['255'],
                         unknown_fraction=counts['255']/int(valid.sum()),
                         artifact_rectangles_xyxy=ARTIFACTS.get(name,[])))
    (DEST/'ai_draft_manifest.json').write_text(json.dumps({
        'status':'UNREVIEWED_AI_ASSISTED_DRAFTS', 'human_completed':False,
        'method':'Conservative brightness/chromaticity seeded region growing; one-pixel unknown boundaries; visually selected artifact rectangles withheld.',
        'independence':'Not independent reference labels. Different from tested B code, but shared colour cues may correlate errors. Human must inspect background and unknown pixels as well as proposed positives.',
        'source':'21 original RGB crops; no yield, frozen predictions, or evaluated segmentation masks used.',
        'generator':str(Path(__file__).relative_to(ROOT)),
        'generator_sha256':sha(Path(__file__)), 'items':rows},indent=2,ensure_ascii=False)+'\n')
    with (DEST/'ai_draft_summary.csv').open('w') as f:
        fields=['image_id','status','valid_pixels','background_pixels','proposed_fiber_pixels','unknown_pixels','unknown_fraction']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    print('Created 21 unverified drafts and previews; completed_masks and human declaration untouched.')

if __name__ == '__main__':
    main()
